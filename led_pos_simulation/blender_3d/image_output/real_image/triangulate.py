import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import TextBox
from collections import OrderedDict
from dataclasses import dataclass
import pickle
import gzip
import cv2
import glob
import os
import matplotlib as mpl
import tkinter as tk
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import json
import matplotlib.ticker as ticker
from enum import Enum, auto
import copy
import re
import subprocess
import cv2
import traceback
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

origin_led_data = np.array([
    [-0.02146761, -0.00343424, -0.01381839],
    [-0.0318701, 0.00568587, -0.01206734],
    [-0.03692925, 0.00930785, 0.00321071],
    [-0.04287211, 0.02691347, -0.00194137],
    [-0.04170018, 0.03609551, 0.01989264],
    [-0.02923584, 0.06186962, 0.0161972],
    [-0.01456789, 0.06295633, 0.03659283],
    [0.00766914, 0.07115411, 0.0206431],
    [0.02992447, 0.05507271, 0.03108736],
    [0.03724313, 0.05268665, 0.01100446],
    [0.04265723, 0.03016438, 0.01624689],
    [0.04222733, 0.0228845, -0.00394005],
    [0.03300807, 0.00371497, 0.00026865],
    [0.03006234, 0.00378822, -0.01297127],
    [0.02000199, -0.00388647, -0.014973]
])
origin_led_dir = np.array([
    [-0.52706841, -0.71386452, -0.46108171],
    [-0.71941994, -0.53832866, -0.43890456],
    [-0.75763735, -0.6234486, 0.19312559],
    [-0.95565641, 0.00827838, -0.29436762],
    [-0.89943476, -0.04857372, 0.43434745],
    [-0.57938915, 0.80424722, -0.13226727],
    [-0.32401356, 0.5869508, 0.74195955],
    [0.14082806, 0.97575588, -0.16753482],
    [0.66436362, 0.41503629, 0.62158335],
    [0.77126662, 0.61174447, -0.17583089],
    [0.90904575, -0.17393345, 0.37865945],
    [0.9435189, -0.10477919, -0.31431419],
    [0.7051038, -0.6950803, 0.14032818],
    [0.67315478, -0.5810967, -0.45737213],
    [0.49720891, -0.70839529, -0.5009585]
])
camera_matrix = [
    [np.array([[712.623, 0.0, 653.448],
               [0.0, 712.623, 475.572],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)],
    [np.array([[716.896, 0.0, 668.902],
               [0.0, 716.896, 460.618],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.07542], [-0.026874], [0.006662], [-0.000775]], dtype=np.float64)]
]
default_dist_coeffs = np.zeros((4, 1))
default_cameraK = np.eye(3).astype(np.float64)

READ = 0
WRITE = 1
SUCCESS = 0
ERROR = -1
DONE = 1
NOT_SET = -1
CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
CV_MIN_THRESHOLD = 100
CV_MAX_THRESHOLD = 255
show_plt = 0

json_file = './blob_area.json'
# 이미지 파일 경로를 지정합니다.
blend_image_l = "./CAMERA_0_blender_test_image.png"
real_image_l = "./left_frame.png"
blend_image_r = "./CAMERA_1_blender_test_image.png"
real_image_r = "./right_frame.png"

# data parsing
CAMERA_INFO = {}
CAMERA_INFO_STRUCTURE = {
    'camera_k': [],
    'led_num': [],
    'points2D': {'greysum': [], 'opencv': [], 'blender': []},
    'points3D': [],
    'rt': {'rvec': [], 'tvec': []},
    'remake_3d': [],
}


class POSE_ESTIMATION_METHOD(Enum):
    # Opencv legacy
    SOLVE_PNP_RANSAC = auto()
    SOLVE_PNP_REFINE_LM = auto()
    SOLVE_PNP_AP3P = auto()
    SOLVE_PNP = auto()
    SOLVE_PNP_RESERVED = auto()
def solvepnp_ransac(*args):
    cam_id = args[0][0]
    points3D = args[0][1]
    points2D = args[0][2]
    camera_k = args[0][3]
    dist_coeff = args[0][4]
    # check assertion
    if len(points3D) != len(points2D):
        print("assertion len is not equal")
        return ERROR, NOT_SET, NOT_SET, NOT_SET

    if len(points2D) < 4:
        print("assertion < 4: ")
        return ERROR, NOT_SET, NOT_SET, NOT_SET

    ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(points3D, points2D,
                                                    camera_k,
                                                    dist_coeff)

    return SUCCESS if ret == True else ERROR, rvecs, tvecs, inliers
def solvepnp_ransac_refineLM(*args):
    cam_id = args[0][0]
    points3D = args[0][1]
    points2D = args[0][2]
    camera_k = args[0][3]
    dist_coeff = args[0][4]
    ret, rvecs, tvecs, inliers = solvepnp_ransac(points3D, points2D, camera_k, dist_coeff)
    # Do refineLM with inliers
    if ret == SUCCESS:
        if not hasattr(cv2, 'solvePnPRefineLM'):
            print('solvePnPRefineLM requires OpenCV >= 4.1.1, skipping refinement')
        else:
            assert len(inliers) >= 3, 'LM refinement requires at least 3 inlier points'
            # refine r_t vector and maybe changed
            cv2.solvePnPRefineLM(points3D[inliers],
                                 points2D[inliers], camera_k, dist_coeff,
                                 rvecs, tvecs)

    return SUCCESS if ret == True else ERROR, rvecs, tvecs, NOT_SET
def solvepnp_AP3P(*args):
    cam_id = args[0][0]
    points3D = args[0][1]
    points2D = args[0][2]
    camera_k = args[0][3]
    dist_coeff = args[0][4]

    # check assertion
    if len(points3D) != len(points2D):
        print("assertion len is not equal")
        return ERROR, NOT_SET, NOT_SET, NOT_SET

    if len(points2D) < 3 or len(points2D) > 4:
        print("assertion ", len(points2D))
        return ERROR, NOT_SET, NOT_SET, NOT_SET

    ret, rvecs, tvecs = cv2.solvePnP(points3D, points2D,
                                     camera_k,
                                     dist_coeff,
                                     flags=cv2.SOLVEPNP_AP3P)

    return SUCCESS if ret == True else ERROR, rvecs, tvecs, NOT_SET
SOLVE_PNP_FUNCTION = {
    POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC: solvepnp_ransac,
    POSE_ESTIMATION_METHOD.SOLVE_PNP_REFINE_LM: solvepnp_ransac_refineLM,
    POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P: solvepnp_AP3P,
}
def pickle_data(rw_mode, path, data):
    import pickle
    import gzip
    try:
        if rw_mode == READ:
            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)
            return data
        elif rw_mode == WRITE:
            with gzip.open(path, 'wb') as f:
                pickle.dump(data, f)
        else:
            print('not support mode')
    except:
        print('file r/w error')
        return ERROR
def rw_json_data(rw_mode, path, data):
    try:
        if rw_mode == READ:
            with open(path, 'r', encoding="utf-8") as rdata:
                json_data = json.load(rdata)
            return json_data
        elif rw_mode == WRITE:
            print(data)
            with open(path, 'w', encoding="utf-8") as wdata:
                json.dump(data, wdata, ensure_ascii=False, indent="\t")
        else:
            print('not support mode')
    except:
        # print('file r/w error')
        return ERROR
def view_camera_infos(frame, text, x, y):
    cv2.putText(frame, text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), lineType=cv2.LINE_AA)
def zoom_factory(ax, base_scale=2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_zlim = ax.get_zlim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
        cur_zrange = (cur_zlim[1] - cur_zlim[0]) * .5
        xdata = event.xdata
        ydata = event.ydata
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange * scale_factor,
                     xdata + cur_xrange * scale_factor])
        ax.set_ylim([ydata - cur_yrange * scale_factor,
                     ydata + cur_yrange * scale_factor])
        ax.set_zlim([(xdata + ydata) / 2 - cur_zrange * scale_factor,
                     (xdata + ydata) / 2 + cur_zrange * scale_factor])
        # force re-draw
        plt.draw()

    # get the figure of interest
    fig = ax.get_figure()
    # attach the call back
    fig.canvas.mpl_connect('scroll_event', zoom_fun)

    # return the function
    return zoom_fun
def find_center(frame, SPEC_AREA):
    x_sum = 0
    t_sum = 0
    y_sum = 0
    g_c_x = 0
    g_c_y = 0
    m_count = 0

    (X, Y, W, H) = SPEC_AREA

    for y in range(Y, Y + H):
        for x in range(X, X + W):
            x_sum += x * frame[y][x]
            t_sum += frame[y][x]
            m_count += 1

    for x in range(X, X + W):
        for y in range(Y, Y + H):
            y_sum += y * frame[y][x]

    if t_sum != 0:
        g_c_x = x_sum / t_sum
        g_c_y = y_sum / t_sum

    if g_c_x == 0 or g_c_y == 0:
        return 0, 0

    result_data_str = f'{g_c_x} ' + f'{g_c_y}'
    # print(result_data_str)

    return g_c_x, g_c_y
def detect_led_lights(image, padding=5):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blob_info = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # 주변 부분을 포함하기 위해 패딩을 적용
        x -= padding
        y -= padding
        w += padding * 2
        h += padding * 2
        blob_info.append([x, y, w, h])

    return blob_info
def add_value(my_dict, key, tag, value):
    compound_key = str(key) + "-" + tag
    my_dict[compound_key] = value  # 이전 값을 삭제하고 새로운 값을 저장합니다.
def draw_bboxes(image, bboxes):
    if len(bboxes) > 0:
        except_pos = 0
        for i, data in enumerate(bboxes):
            (x, y, w, h) = data['bbox']
            IDX = data['idx']
            except_pos += 1
            if except_pos == len(bboxes) / 2:
                color = (255, 255, 255)
                line_width = 1
                except_pos = 0
            else:
                color = (0, 255, 0)
                line_width = 2

            cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, line_width,
                          1)
            view_camera_infos(image, ''.join([f'{IDX}']), int(x), int(y) - 10)
def blob_area_set(index, image, tag):
    bboxes = []
    json_data = rw_json_data(READ, json_file, None)
    compound_key = str(index) + "-" + tag
    if json_data != ERROR:
        if compound_key in json_data:
            bboxes = json_data[compound_key]['bboxes']
    else:
        json_data = OrderedDict()
    print(bboxes)
    while True:
        draw_img = image.copy()
        _, img_filtered = cv2.threshold(draw_img, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
        IMG_GRAY = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
        draw_bboxes(draw_img, bboxes)
        key = cv2.waitKey(1)
        if key == ord('a'):
            cv2.imshow(f"{compound_key}", IMG_GRAY)
            bbox = cv2.selectROI(f"{compound_key}", IMG_GRAY)
            while True:
                inputs = input('input led number: ')
                if inputs.isdigit():
                    input_number = int(inputs)
                    if input_number in range(0, 1000):
                        (x, y, w, h) = bbox
                        print('label number ', input_number)
                        print('bbox ', bbox)
                        if x >= CAP_PROP_FRAME_WIDTH:
                            cam_id = 1
                        else:
                            cam_id = 0
                        bboxes.append({'idx': input_number, 'bbox': bbox, 'id': cam_id})
                        break
                elif cv2.waitKey(1) == ord('q'):
                    bboxes.clear()
                    break
        elif key == ord('c'):
            print('clear area')
            bboxes.clear()

        elif key == ord('s'):
            print('save blob area')
            add_value(json_data, index, tag, {'bboxes': bboxes})
            # Write json data
            rw_json_data(WRITE, json_file, json_data)

        elif key & 0xFF == 27:
            print('ESC pressed')
            cv2.destroyAllWindows()
            return ERROR
        elif key == ord('n'):
            print('go next step')
            break

        cv2.imshow(f"{compound_key}", draw_img)
def remake_3d_point(camera_k_0, camera_k_1, RT_0, RT_1, BLOB_0, BLOB_1):
    l_rotation, _ = cv2.Rodrigues(RT_0['rvec'])
    r_rotation, _ = cv2.Rodrigues(RT_1['rvec'])
    l_projection = np.dot(camera_k_0, np.hstack((l_rotation, RT_0['tvec'])))
    r_projection = np.dot(camera_k_1, np.hstack((r_rotation, RT_1['tvec'])))
    l_blob = np.reshape(BLOB_0, (1, len(BLOB_0), 2))
    r_blob = np.reshape(BLOB_1, (1, len(BLOB_1), 2))
    triangulation = cv2.triangulatePoints(l_projection, r_projection,
                                          l_blob, r_blob)
    homog_points = triangulation.transpose()
    get_points = cv2.convertPointsFromHomogeneous(homog_points)
    return get_points
def calculation(key, bboxes, IMG_GRAY, *args):
    DEBUG = 0
    draw_img = copy.deepcopy(IMG_GRAY)
    data_key = copy.deepcopy(key)
    ax1 = args[0][0]
    ax2 = args[0][1]
    ax3 = args[0][2]
    CAMERA_INFO[f"{data_key}_0"] = copy.deepcopy(CAMERA_INFO_STRUCTURE)
    CAMERA_INFO[f"{data_key}_1"] = copy.deepcopy(CAMERA_INFO_STRUCTURE)
    ax2.set_title(f'2D Image and Projection of GreySum {data_key}')
    ax2.imshow(draw_img, cmap='gray')

    LED_NUMBER = []
    if len(bboxes) > 0:
        for i, data in enumerate(bboxes):
            (x, y, w, h) = data['bbox']
            IDX = int(data['idx'])
            cam_id = int(data['id'])
            cx, cy = find_center(IMG_GRAY, (x, y, w, h))
            if cam_id == 1:
                cx -= CAP_PROP_FRAME_WIDTH
                LED_NUMBER.append(IDX)
            CAMERA_INFO[f"{data_key}_{cam_id}"]['points2D']['greysum'].append([cx, cy])
            CAMERA_INFO[f"{data_key}_{cam_id}"]['points3D'].append(origin_led_data[IDX])

    if len(LED_NUMBER) >= 5:
        METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC
    else:
        METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P
    if DEBUG == 1:
        print('data key', key)
        print(CAMERA_INFO[f"{data_key}_0"]['points2D'])
        print(CAMERA_INFO[f"{data_key}_1"]['points2D'])
    for keys, cam_data in CAMERA_INFO.items():
        if data_key not in keys:
            continue
        cam_id = int(keys.split('_')[1])
        cam_data['led_num'] = LED_NUMBER
        points2D = np.array(cam_data['points2D']['greysum'], dtype=np.float64)
        points3D = np.array(cam_data['points3D'], dtype=np.float64)
        if DEBUG == 1:
            print('point_3d\n', points3D)
            print('point_2d\n', points2D)

        INPUT_ARRAY = [
            cam_id,
            points3D,
            points2D,
            camera_matrix[cam_id][0],
            camera_matrix[cam_id][1]
        ]

        ret, rvec, tvec, inliers = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
        if DEBUG == 1:
            print('RT from OpenCV SolvePnP')
            print('rvec', rvec)
            print('tvec', tvec)
        cam_data['rt']['rvec'] = rvec
        cam_data['rt']['tvec'] = tvec

        greysum_points = points2D.reshape(-1, 2)
        # 3D 점들을 2D 이미지 평면에 투영
        image_points, _ = cv2.projectPoints(points3D, rvec, tvec, camera_matrix[cam_id][0], camera_matrix[cam_id][1])
        image_points = image_points.reshape(-1, 2)

        if 'BL' in data_key:
            color = 'blue'
        else:
            color = 'red'
        if cam_id == 0:
            ax2.scatter(greysum_points[:, 0], greysum_points[:, 1], c='black', alpha=0.5, label='GreySum', s=7)
            ax2.scatter(image_points[:, 0], image_points[:, 1], c=color, alpha=0.5, label='OpenCV', s=7)
        else:
            ax2.scatter(greysum_points[:, 0] + CAP_PROP_FRAME_WIDTH, greysum_points[:, 1], c='black', alpha=0.7, label='GreySum', s=1)
            ax2.scatter(image_points[:, 0] + CAP_PROP_FRAME_WIDTH, image_points[:, 1], c=color, alpha=0.7, label='OpenCV', s=1)

    remake_3d = remake_3d_point(camera_matrix[0][0], camera_matrix[1][0],
                                CAMERA_INFO[f"{data_key}_0"]['rt'],
                                CAMERA_INFO[f"{data_key}_1"]['rt'],
                                CAMERA_INFO[f"{data_key}_0"]['points2D']['greysum'],
                                CAMERA_INFO[f"{data_key}_1"]['points2D']['greysum']).reshape(-1, 3)
    CAMERA_INFO[f"{data_key}_0"]['remake_3d'] = points3D.reshape(-1, 3)
    CAMERA_INFO[f"{data_key}_1"]['remake_3d'] = points3D.reshape(-1, 3)

    origin_pts = np.array(origin_led_data).reshape(-1, 3)

    ax1.scatter(origin_pts[:, 0], origin_pts[:, 1], origin_pts[:, 2], color='black', alpha=0.5, marker='o',
                s=10)
    ax1.scatter(0, 0, 0, marker='o', color='k', s=20)
    ax1.set_xlim([-0.1, 0.1])
    ax1.set_xlabel('X')
    ax1.set_ylim([-0.1, 0.1])
    ax1.set_ylabel('Y')
    ax1.set_zlim([-0.1, 0.1])
    ax1.set_zlabel('Z')
    scale = 1.5
    f = zoom_factory(ax1, base_scale=scale)

    dist_remake_3d = np.linalg.norm(points3D.reshape(-1, 3) - remake_3d, axis=1)
    ax3.bar(range(len(dist_remake_3d)), dist_remake_3d, width=0.4)
    ax3.set_title(f"Distance between origin_pts and remake {data_key}")
    ax3.set_xlabel('LEDS')
    ax3.set_ylabel('Distance')
    ax3.set_xticks(range(len(dist_remake_3d)))
    ax3.set_xticklabels(LED_NUMBER)
    ax3.set_yscale('log')

    if DEBUG == 1:
        print('remake_3d\n', remake_3d)
        print('origin_pts\n', points3D.reshape(-1, 3))
        print('dist_remake_3d\n', dist_remake_3d)
def trianglute_test(index, img_bl, img_re):
    try:
        root = tk.Tk()
        width_px = root.winfo_screenwidth()
        height_px = root.winfo_screenheight()

        # 모니터 해상도에 맞게 조절
        mpl.rcParams['figure.dpi'] = 120  # DPI 설정
        monitor_width_inches = width_px / mpl.rcParams['figure.dpi']  # 모니터 너비를 인치 단위로 변환
        monitor_height_inches = height_px / mpl.rcParams['figure.dpi']  # 모니터 높이를 인치 단위로 변환

        fig = plt.figure(figsize=(monitor_width_inches, monitor_height_inches), num='LED Position FinDer')

        # 2:1 비율로 큰 그리드 생성
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

        # 왼쪽 그리드에 subplot 할당
        ax1 = plt.subplot(gs[0], projection='3d')

        # 오른쪽 그리드를 위에는 2개, 아래는 3개로 분할
        gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], height_ratios=[1, 1])

        # 분할된 오른쪽 그리드의 위쪽에 subplot 할당
        gs_sub_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_sub[0])
        ax2 = plt.subplot(gs_sub_top[0])
        ax3 = plt.subplot(gs_sub_top[1])

        # 분할된 오른쪽 그리드의 아래쪽에 subplot 할당
        gs_sub_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_sub[1])
        ax4 = plt.subplot(gs_sub_bottom[0])
        ax5 = plt.subplot(gs_sub_bottom[1])

        img_bl = img_bl.copy()
        img_re = img_re.copy()
        json_data = rw_json_data(READ, json_file, None)
        bboxes_bl = json_data[f"{index}-BL"]['bboxes']
        bboxes_re = json_data[f"{index}-RE"]['bboxes']
        _, img_bl_filtered = cv2.threshold(img_bl, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
        IMG_GRAY_BL = cv2.cvtColor(img_bl_filtered, cv2.COLOR_BGR2GRAY)
        _, img_re_filtered = cv2.threshold(img_re, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
        IMG_GRAY_RE = cv2.cvtColor(img_re_filtered, cv2.COLOR_BGR2GRAY)
        draw_bboxes(img_bl, bboxes_bl)
        draw_bboxes(img_re, bboxes_re)

        # cv2.imshow(f"{index}-BL", img_bl)
        # cv2.imshow(f"{index}-RE", img_re)
        calculation(f"{index}-BL", bboxes_bl, IMG_GRAY_BL, [ax1, ax2, ax4])
        calculation(f"{index}-RE", bboxes_re, IMG_GRAY_RE, [ax1, ax3, ax5])

        if show_plt:
            plt.show()
        key = cv2.waitKey(0)

    except:
        traceback.print_exc()
        return ERROR
def triangulate_test():
    curr_index = 0
    prev_index = -1
    while True:
        B_img_l = cv2.imread(blend_image_l)
        B_img_r = cv2.imread(blend_image_r)
        R_img_l = cv2.imread(real_image_l)
        R_img_r = cv2.imread(real_image_r)
        STACK_FRAME_B = np.hstack((B_img_l, B_img_r))
        STACK_FRAME_R = np.hstack((R_img_l, R_img_r))
        if prev_index != curr_index:
            if blob_area_set(curr_index, STACK_FRAME_B, 'BL') == ERROR:
                break
            if blob_area_set(curr_index, STACK_FRAME_R, 'RE') == ERROR:
                break
            prev_index = curr_index
        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            print('ESC pressed')
            break
        elif key == ord('c'):
            print('capture')
            if trianglute_test(curr_index, STACK_FRAME_B, STACK_FRAME_R) == ERROR:
                break
            curr_index += 1
            cv2.destroyAllWindows()
        elif key == ord('n'):
            curr_index += 1
            cv2.destroyAllWindows()
        else:
            print('capture mode')

    cv2.destroyAllWindows()
    file = './camera_info.pickle'
    data = OrderedDict()
    data['CAMERA_INFO'] = CAMERA_INFO
    ret = pickle_data(WRITE, file, data)
    if ret != ERROR:
        print('data saved')
def test_result():
    pickle_file = './camera_info.pickle'
    data = pickle_data(READ, pickle_file, None)
    CAMERA_INFO = data['CAMERA_INFO']

    for key, value in CAMERA_INFO.items():
        print('############################')
        print('key', key)
        print(value)
        print('############################')
        print('\n')

    # 모델과 카메라 ID별로 rvec, tvec 값 저장할 딕셔너리 초기화
    rvecs = {}
    tvecs = {}

    # 평균과 표준편차를 저장할 데이터프레임 생성
    df = pd.DataFrame(columns=['model_cam_id', 'type', 'mean', 'std'])
    for key, value in CAMERA_INFO.items():
        model_cam_id = key.split('-')[1]  # 'BL_0', 'BL_1', 'RE_0', 'RE_1' 등 추출
        rvec = np.linalg.norm(value['rt']['rvec'])  # rvec 벡터의 길이 계산
        tvec = np.linalg.norm(value['rt']['tvec'])  # tvec 벡터의 길이 계산

        if model_cam_id not in rvecs:
            rvecs[model_cam_id] = []
        if model_cam_id not in tvecs:
            tvecs[model_cam_id] = []

        rvecs[model_cam_id].append(rvec)
        tvecs[model_cam_id].append(tvec)

    # 각 모델과 카메라 ID에 대해 rvec, tvec의 평균과 표준편차 계산
    for idx, model_cam_id in enumerate(rvecs.keys()):
        rvec_arr = np.array(rvecs[model_cam_id])
        tvec_arr = np.array(tvecs[model_cam_id])

        rvec_mean = np.mean(rvec_arr)
        tvec_mean = np.mean(tvec_arr)

        rvec_std = np.std(rvec_arr)
        tvec_std = np.std(tvec_arr)

        print(f'model and camera id: {model_cam_id}')
        print(f'rvec mean: {rvec_mean}, rvec std: {rvec_std}')
        print(f'tvec mean: {tvec_mean}, tvec std: {tvec_std}')
        print('\n')
        df.loc[2 * idx] = [model_cam_id, 'rvec', rvec_mean, rvec_std]
        df.loc[2 * idx + 1] = [model_cam_id, 'tvec', tvec_mean, tvec_std]

    # 그래프 그리기
    plt.figure(figsize=(10, 6))  # 그래프 사이즈 설정
    sns.barplot(x='model_cam_id', y='std', hue='type', data=df, palette=['#53a4b1', '#c06343'],
                capsize=0.1)  # 바 그래프 그리기
    plt.title('Standard Deviation of rvec & tvec for each model')  # 그래프 제목 설정
    plt.show()  # 그래프 보여주기


if __name__ == "__main__":
    for i, leds in enumerate(origin_led_data):
        print(f"{i}, {leds}")
    # triangulate_test()
    test_result()

