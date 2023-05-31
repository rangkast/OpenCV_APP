import numpy as np
import matplotlib.pyplot as plt
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
        'points2D': {'greysum': [], 'opencv': [], 'blender': []},
        'points3D': [],
        'opencv_rt': {'rvec': [], 'tvec': []},
        'blender_rt': {'rvec': [], 'tvec': []},
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
    print(result_data_str)

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
def calculation(key, bboxes, IMG_GRAY):
    CAMERA_INFO[f"{key}_0"] = copy.deepcopy(CAMERA_INFO_STRUCTURE)
    CAMERA_INFO[f"{key}_1"] = copy.deepcopy(CAMERA_INFO_STRUCTURE)

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
                print('key', f"{key}_{cam_id}")
            CAMERA_INFO[f"{key}_{cam_id}"]['points2D']['greysum'].append([cx, cy])
            CAMERA_INFO[f"{key}_{cam_id}"]['points3D'].append(origin_led_data[IDX])

    METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P

    print(CAMERA_INFO[f"{key}_0"]['points2D'])
    print(CAMERA_INFO[f"{key}_1"]['points2D'])
    for key, cam_data in CAMERA_INFO.items():
        cam_id = int(key.split('_')[1])
        print('CAM id', cam_id, cam_data)
        points2D = np.array(cam_data['points2D']['greysum'], dtype=np.float64)
        points3D = np.array(cam_data['points3D'], dtype=np.float64)
        print('point_3d\n', points3D)
        print('point_2d\n', points2D)
        #
        greysum_points = points2D.reshape(-1, 2)
        for g_point in greysum_points:
            cv2.circle(IMG_GRAY, (int(g_point[0]), int(g_point[1])), 1, (0, 0, 0), -1)

        dist_coeff = default_dist_coeffs
        INPUT_ARRAY = [
            cam_id,
            points3D,
            points2D,
            camera_matrix[cam_id][0],
            # camera_matrix[cam_id][1]
            dist_coeff
        ]

        ret, rvec, tvec, inliers = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)

        print('RT from OpenCV SolvePnP')
        print('rvec', rvec)
        print('tvec', tvec)


def trianglute_test(index, img_bl, img_re):
    try:
        img_bl = img_bl.copy()
        img_re = img_re.copy()
        json_data = rw_json_data(READ, json_file, None)
        bboxes_bl = json_data[f"{index}-BL"]['bboxes']
        bboxes_re = json_data[f"{index}-RE"]['bboxes']
        _, img_filtered = cv2.threshold(img_bl, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
        IMG_GRAY_BL = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
        _, img_filtered = cv2.threshold(img_re, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
        IMG_GRAY_RE = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
        draw_bboxes(img_bl, bboxes_bl)
        draw_bboxes(img_re, bboxes_re)

        # cv2.imshow(f"{index}-BL", img_bl)
        # cv2.imshow(f"{index}-RE", img_re)
        calculation(f"{index}-BL", bboxes_bl, IMG_GRAY_BL)
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
        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            print('ESC pressed')
            cv2.destroyAllWindows()
            return ERROR
        elif key == ord('c'):
            print('go next step')
            if trianglute_test(curr_index, STACK_FRAME_B, STACK_FRAME_R) == ERROR:
                cv2.destroyAllWindows()
                return ERROR
            curr_index += 1
        else:
            prev_index = curr_index


if __name__ == "__main__":
    for i, leds in enumerate(origin_led_data):
        print(f"{i}, {leds}")
    triangulate_test()
