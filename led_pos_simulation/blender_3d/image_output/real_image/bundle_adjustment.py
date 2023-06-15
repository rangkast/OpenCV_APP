import numpy as np
import random
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
import itertools
from typing import List, Dict
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import functools

READ = 0
WRITE = 1
SUCCESS = 0
ERROR = -1
DONE = 1
NOT_SET = -1
CAM_ID = 0
undistort = 0
# 설계값
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

target_led_data = np.array([
     [-0.01984326, -0.004046,   -0.01434656],
     [-0.03294307,  0.00655128, -0.01436888],
     [-0.03518444,  0.00854664,  0.00352975],
     [-0.04312148,  0.02837558, -0.00400151],
     [-0.0420226,  0.03571146,  0.02102641],
     [-0.03033573,  0.06169719,  0.01531934],
     [-0.01452568,  0.06353915,  0.03549221],
     [ 0.00881386,  0.0720557,   0.02114559],
     [ 0.03082533,  0.05438898,  0.03096447],
     [ 0.03630736,  0.05241876,  0.01153482],
     [ 0.04196557,  0.02976763,  0.01555972],
     [ 0.04138212,  0.02221325, -0.00395271],
     [ 0.03189076,  0.00394939,  0.00192845],
     [ 0.03080438,  0.00359638, -0.0138589 ],
     [ 0.01925483, -0.00219402, -0.01492219],
])

target_pose_led_data = np.array([
     [-0.01108217, -0.00278021, -0.01373098],
     [-0.02405356,  0.00777868, -0.0116913 ],
     [-0.02471722,  0.00820648,  0.00643996],
     [-0.03312733,  0.02861635,  0.0013793 ],
     [-0.02980387,  0.03374299,  0.02675826],
     [-0.0184596,  0.06012725,  0.02233215],
     [-0.00094422,  0.06020401,  0.04113377],
     [ 0.02112556,  0.06993855,  0.0256014 ],
     [ 0.04377158,  0.05148328,  0.03189337],
     [ 0.04753083,  0.05121397,  0.01196245],
     [ 0.0533449,  0.02829823, 0.01349697],
     [ 0.05101214,  0.02247323, -0.00647229],
     [ 0.04192879,  0.00376628, -0.00139432],
     [ 0.03947314,  0.00479058, -0.01699771],
     [ 0.02783124, -0.00088511, -0.01754906],
])

# # Set the seed for Python's random module.
# random.seed(1)
# # Set the seed for NumPy's random module.
# np.random.seed(1)
# noise_std_dev = 0.001  # Noise standard deviation. Adjust this value to your needs.
# # Generate noise with the same shape as the original data.
# noise = np.random.normal(scale=noise_std_dev, size=origin_led_data.shape)
# # Add noise to the original data.
# noisy_led_data = origin_led_data + noise

camera_matrix = [
    [np.array([[712.623, 0.0, 653.448],
               [0.0, 712.623, 475.572],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)],
]
default_dist_coeffs = np.zeros((4, 1))
default_cameraK = np.eye(3).astype(np.float64)
CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
CV_MIN_THRESHOLD = 100
CV_MAX_THRESHOLD = 255
# data parsing
CAMERA_INFO = {}
CAMERA_INFO_STRUCTURE = {
    'camera_calibration': {'camera_k': [], 'dist_coeff': []},
    'led_num': [],
    'points2D': {'greysum': [], 'opencv': [], 'blender': []},
    'points2D_U': {'greysum': [], 'opencv': [], 'blender': []},
    'points3D': [],
    'points3D_target': [],
    'BLENDER': {'rt': {'rvec': [], 'tvec': []}, 'remake_3d': [],  'distance_o': [], 'distance_t': []},
    'OPENCV': {'rt': {'rvec': [], 'tvec': []}, 'remake_3d': [], 'distance_o': [], 'distance_t': []},
}
path = './bundle_area.json'



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
        # Apply padding for the bounding box
        x -= padding
        y -= padding
        w += padding * 2
        h += padding * 2
        blob_info.append((x, y, w, h))

    return blob_info
def point_in_bbox(x, y, bbox):
    return bbox[0] <= x <= bbox[0] + bbox[2] and bbox[1] <= y <= bbox[1] + bbox[3]
def draw_blobs_and_ids(frame, blobs, trackers):
    for bbox in blobs:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 1, 1)
    for id, data in trackers.items():
        cv2.putText(frame, f'{id}',  (int(data['bbox'][0]), int(data['bbox'][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
def rw_json_data(rw_mode, path, data=None):
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
        return ERROR
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
def init_trackers(trackers, frame):
    for id, data in trackers.items():
        tracker = cv2.TrackerCSRT_create()
        ok = tracker.init(frame, data['bbox'])
        data['tracker'] = tracker
def click_event(event, x, y, flags, param, frame_0, blob_area_0, trackers):
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, bbox in enumerate(blob_area_0):
            if point_in_bbox(x, y, bbox):
                id = input('Please enter ID for this bbox: ')
                trackers[id] = {'bbox': bbox, 'tracker': None}
                draw_blobs_and_ids(frame_0, blob_area_0, trackers)
def read_camera_log(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # print(lines)
    camera_params = {}
    for line in lines:
        parts = line.split(',')
        frame = int(parts[0].split(':')[1].strip())
        rvec = list(map(float, parts[1].split(':')[1].strip()[1:-1].split()))
        tvec = list(map(float, parts[2].split(':')[1].strip()[1:-1].split()))
        camera_params[frame] = (rvec, tvec)


    return camera_params
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
def gathering_data():
    VIDEO = 0
    SHOW_IMG_ONLY = 1
    # Read the first frame
    # Read the video
    if VIDEO:
        cap_0 = cv2.VideoCapture('./CAMERA_0_blender_test.mkv')
        ret_0, frame_0 = cap_0.read()
        if not ret_0:
            print("Cannot read the video")
            cap_0.release()
            cv2.destroyAllWindows()
            exit()
    else:
        # camera_params = read_camera_log('C:/Users/user/tmp/render/camera_log.txt')
        # image_files = sorted(glob.glob('C:/Users/user/tmp/render/*.png'))

        camera_params = read_camera_log('./tmp/render/camera_log.txt')
        image_files = sorted(glob.glob('./tmp/render/*.png'))

        frame_0 = cv2.imread(image_files[0])
        if frame_0 is None:
            print("Cannot read the first image")
            cv2.destroyAllWindows()
            exit()

    _, frame_0 = cv2.threshold(cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                               cv2.THRESH_TOZERO)
    draw_frame = frame_0.copy()
    # Try to load previous data
    prev_data = rw_json_data(READ, path)
    if prev_data != ERROR:
        trackers = prev_data
        init_trackers(trackers, frame_0)  # Create new tracker instances
    else:
        trackers = {}
    blob_area_0 = detect_led_lights(draw_frame, 5)
    cv2.namedWindow('image')
    partial_click_event = functools.partial(click_event, frame_0=frame_0, blob_area_0=blob_area_0, trackers=trackers)
    cv2.setMouseCallback('image', partial_click_event)
    # Wait for user to click and assign IDs
    while True:
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('c'):  # Reset all IDs and trackers
            trackers = {}
            draw_frame = frame_0.copy()
        elif key & 0xFF == ord('s'):  # Save data to json file
            # Save a copy of the data without the tracker instances
            save_data = {id: {'bbox': data['bbox']} for id, data in trackers.items()}
            rw_json_data(WRITE, path, save_data)
            init_trackers(trackers, frame_0)
        elif key & 0xFF == 27:
            print('ESC pressed')
            cv2.destroyAllWindows()
            if VIDEO:
                cap_0.release()
            return ERROR

        draw_blobs_and_ids(draw_frame, blob_area_0, trackers)
        cv2.imshow('image', draw_frame)
    cv2.destroyAllWindows()

    frame_cnt = 0
    if VIDEO:
    # read the video
        while True:
            # Read a new frame
            ret_0, frame_0 = cap_0.read()
            if not ret_0:
                break
    # read the image
    else:
        while frame_cnt < len(image_files):
            # Read a new frame
            frame_0 = cv2.imread(image_files[frame_cnt])
            if frame_0 is None:
                break
            draw_frame = frame_0.copy()

            _, frame_0 = cv2.threshold(cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                       cv2.THRESH_TOZERO)

            # Draw the filename at the top-right corner
            if VIDEO == 0:
                filename = os.path.basename(image_files[frame_cnt])
                cv2.putText(draw_frame, filename, (draw_frame.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                            1)
            # Update trackers
            tracking_failed = []
            LED_NUMBER = []
            CAMERA_INFO[f"{frame_cnt}"] = copy.deepcopy(CAMERA_INFO_STRUCTURE)

            blob_area_0 = detect_led_lights(frame_0, 5)
            for bbox in blob_area_0:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(draw_frame, p1, p2, (255, 255, 255), 1, 1)
            for id, data in trackers.items():
                ok, bbox = data['tracker'].update(frame_0)
                IDX = int(id)
                # Draw bounding box
                if ok:
                    (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 0), 1, 1)
                    cv2.putText(draw_frame, f'{IDX}',  (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                    cx, cy = find_center(frame_0, (x, y, w, h))
                    LED_NUMBER.append(IDX)
                    CAMERA_INFO[f"{frame_cnt}"]['points2D']['greysum'].append([cx, cy])
                    CAMERA_INFO[f"{frame_cnt}"]['points3D'].append(origin_led_data[IDX])
                    CAMERA_INFO[f"{frame_cnt}"]['points3D_target'].append(target_led_data[IDX])
                else:
                    if id not in tracking_failed:
                        tracking_failed.append(id)
                    cv2.putText(draw_frame, f"Tracking failure detected {', '.join(map(str, tracking_failed))}", (100, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
            if SHOW_IMG_ONLY == 0:
                if len(LED_NUMBER) >= 5:
                    METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC
                else:
                    METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P

                CAMERA_INFO[f"{frame_cnt}"]['led_num'] = LED_NUMBER
                points2D = np.array(CAMERA_INFO[f"{frame_cnt}"]['points2D']['greysum'], dtype=np.float64)
                # TEST
                points3D = np.array(CAMERA_INFO[f"{frame_cnt}"]['points3D'], dtype=np.float64)

                CAMERA_INFO[f"{frame_cnt}"]['points2D_U']['greysum'] = np.array(cv2.undistortPoints(points2D, camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][1])).reshape(-1, 2)

                INPUT_ARRAY = [
                    CAM_ID,
                    points3D,
                    points2D if undistort == 0 else CAMERA_INFO[f"{frame_cnt}"]['points2D_U']['greysum'],
                    camera_matrix[CAM_ID][0] if undistort == 0 else default_cameraK,
                    camera_matrix[CAM_ID][1] if undistort == 0 else default_dist_coeffs
                ]
                ret, rvec, tvec, inliers = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
                CAMERA_INFO[f"{frame_cnt}"]['OPENCV']['rt']['rvec'] = rvec
                CAMERA_INFO[f"{frame_cnt}"]['OPENCV']['rt']['tvec'] = tvec
                if VIDEO == 0:
                    brvec, btvec = camera_params[frame_cnt + 1]
                    print('brvec:', brvec)
                    print('btvec:', btvec)
                    CAMERA_INFO[f"{frame_cnt}"]['BLENDER']['rt']['rvec'] = np.array(brvec).reshape(-1,1)
                    CAMERA_INFO[f"{frame_cnt}"]['BLENDER']['rt']['tvec'] = np.array(btvec).reshape(-1,1)
                    image_points, _ = cv2.projectPoints(points3D,
                                                        np.array(brvec),
                                                        np.array(btvec),
                                                        camera_matrix[CAM_ID][0],
                                                        camera_matrix[CAM_ID][1])
                    image_points = image_points.reshape(-1, 2)
                    for point in image_points:
                        # 튜플 형태로 좌표 변환
                        pt = (int(point[0]), int(point[1]))
                        cv2.circle(draw_frame, pt, 1, (255, 0, 0), -1)

                if frame_cnt > 0:
                    # CALC Using Blender RT
                    if undistort == 0:
                        remake_3d = remake_3d_point(camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][0],
                                                    CAMERA_INFO['0']['BLENDER']['rt'],
                                                    CAMERA_INFO[f"{frame_cnt}"]['BLENDER']['rt'],
                                                    CAMERA_INFO['0']['points2D']['greysum'],
                                                    CAMERA_INFO[f"{frame_cnt}"]['points2D']['greysum']).reshape(-1, 3)
                    else:
                        remake_3d = remake_3d_point(default_cameraK, default_cameraK,
                                                    CAMERA_INFO['0']['BLENDER']['rt'],
                                                    CAMERA_INFO[f"{frame_cnt}"]['BLENDER']['rt'],
                                                    CAMERA_INFO['0']['points2D_U']['greysum'],
                                                    CAMERA_INFO[f"{frame_cnt}"]['points2D_U']['greysum']).reshape(-1, 3)

                    CAMERA_INFO[f"{frame_cnt}"]['BLENDER']['remake_3d'] = remake_3d.reshape(-1, 3)
                    TARGET = np.array(CAMERA_INFO[f"{frame_cnt}"]['points3D_target'], dtype=np.float64)
                    dist_diff_t = np.linalg.norm(TARGET.reshape(-1, 3) - CAMERA_INFO[f"{frame_cnt}"]['BLENDER']['remake_3d'], axis=1)
                    CAMERA_INFO[f"{frame_cnt}"]['BLENDER']['distance_t'] = dist_diff_t
                    dist_diff_o = np.linalg.norm(points3D.reshape(-1, 3) - CAMERA_INFO[f"{frame_cnt}"]['BLENDER']['remake_3d'], axis=1)
                    CAMERA_INFO[f"{frame_cnt}"]['BLENDER']['distance_o'] = dist_diff_o
                    if frame_cnt == 1:
                        CAMERA_INFO['0']['BLENDER']['remake_3d'] = remake_3d.reshape(-1, 3)
                        CAMERA_INFO['0']['BLENDER']['distance_t'] = dist_diff_t
                        CAMERA_INFO['0']['BLENDER']['distance_o'] = dist_diff_o

                    # CALC Using OpenCV RT
                    if undistort == 0:
                        remake_3d = remake_3d_point(camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][0],
                                                    CAMERA_INFO['0']['OPENCV']['rt'],
                                                    CAMERA_INFO[f"{frame_cnt}"]['OPENCV']['rt'],
                                                    CAMERA_INFO['0']['points2D']['greysum'],
                                                    CAMERA_INFO[f"{frame_cnt}"]['points2D']['greysum']).reshape(-1, 3)
                    else:
                        remake_3d = remake_3d_point(default_cameraK, default_cameraK,
                                                    CAMERA_INFO['0']['OPENCV']['rt'],
                                                    CAMERA_INFO[f"{frame_cnt}"]['OPENCV']['rt'],
                                                    CAMERA_INFO['0']['points2D_U']['greysum'],
                                                    CAMERA_INFO[f"{frame_cnt}"]['points2D_U']['greysum']).reshape(-1, 3)

                    CAMERA_INFO[f"{frame_cnt}"]['OPENCV']['remake_3d'] = remake_3d.reshape(-1, 3)
                    TARGET = np.array(CAMERA_INFO[f"{frame_cnt}"]['points3D_target'], dtype=np.float64)
                    dist_diff_t = np.linalg.norm(TARGET.reshape(-1, 3) - CAMERA_INFO[f"{frame_cnt}"]['OPENCV']['remake_3d'], axis=1)
                    CAMERA_INFO[f"{frame_cnt}"]['OPENCV']['distance_t'] = dist_diff_t
                    dist_diff_o = np.linalg.norm(points3D.reshape(-1, 3) - CAMERA_INFO[f"{frame_cnt}"]['OPENCV']['remake_3d'], axis=1)
                    CAMERA_INFO[f"{frame_cnt}"]['OPENCV']['distance_o'] = dist_diff_o
                    if frame_cnt == 1:
                        CAMERA_INFO['0']['OPENCV']['remake_3d'] = remake_3d.reshape(-1, 3)
                        CAMERA_INFO['0']['OPENCV']['distance_t'] = dist_diff_t
                        CAMERA_INFO['0']['OPENCV']['distance_o'] = dist_diff_o

            print('camera_info\n', CAMERA_INFO[f"{frame_cnt}"])

            # Display result
            cv2.imshow("Tracking", draw_frame)

            frame_cnt += 1
            key = cv2.waitKey(1)
            # Exit if ESC key is
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == 27:
                print('ESC pressed')
                if VIDEO:
                    cap_0.release()
                cv2.destroyAllWindows()
                return ERROR
    if VIDEO:
        cap_0.release()
    cv2.destroyAllWindows()
    file = 'bundle_adjustment.pickle'
    data = OrderedDict()
    data['CAMERA_INFO'] = CAMERA_INFO
    ret = pickle_data(WRITE, file, data)
    if ret != ERROR:
        print('data saved')

    root = tk.Tk()
    width_px = root.winfo_screenwidth()
    height_px = root.winfo_screenheight()

    # 모니터 해상도에 맞게 조절
    mpl.rcParams['figure.dpi'] = 120  # DPI 설정
    monitor_width_inches = width_px / mpl.rcParams['figure.dpi']  # 모니터 너비를 인치 단위로 변환
    monitor_height_inches = height_px / mpl.rcParams['figure.dpi']  # 모니터 높이를 인치 단위로 변환

    fig = plt.figure(figsize=(monitor_width_inches, monitor_height_inches), num='Camera Simulator')

    plt.rcParams.update({'font.size': 7})
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax1 = plt.subplot(gs[0], projection='3d')
    ax2 = plt.subplot(gs[1])
    led_number = len(origin_led_data)
    ax2.set_title('distance')
    ax2.set_xlim([0, led_number])  # x축을 LED 번호의 수 만큼 설정합니다. 이 경우 14개로 설정됩니다.
    ax2.set_xticks(range(led_number))  # x축에 표시되는 눈금을 LED 번호의 수 만큼 설정합니다.
    ax2.set_xticklabels(range(led_number))  # x축에 표시되는 눈금 라벨을 LED 번호의 수 만큼 설정합니다.

    # ax1 설정
    origin_pts = np.array(origin_led_data).reshape(-1, 3)
    target_pts = np.array(target_led_data).reshape(-1, 3)
    ax1.set_title('3D plot')
    ax1.scatter(origin_pts[:, 0], origin_pts[:, 1], origin_pts[:, 2], color='black', alpha=0.5, marker='o', s=13)
    ax1.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2], color='red', alpha=0.5, marker='o', s=13)
    ax1.scatter(0, 0, 0, marker='o', color='k', s=20)
    ax1.set_xlim([-0.1, 0.1])
    ax1.set_xlabel('X')
    ax1.set_ylim([-0.1, 0.1])
    ax1.set_ylabel('Y')
    ax1.set_zlim([-0.1, 0.1])
    ax1.set_zlabel('Z')
    scale = 1.5
    f = zoom_factory(ax1, base_scale=scale)  # 이부분은 ax1이 zoom in, zoom out 기능을 가지게 해주는 코드입니다.
    handles1 = []
    handles2 = []
    for key, cam_data in CAMERA_INFO.items():
        led_index = np.array(cam_data['led_num'])

        handle = ax1.scatter(cam_data['BLENDER']['remake_3d'][:, 0], cam_data['BLENDER']['remake_3d'][:, 1],
                             cam_data['BLENDER']['remake_3d'][:, 2], color='blue', alpha=0.5, marker='o', s=10,
                             label='BLENDER')
        handles1.append(handle)

        handle = ax1.scatter(cam_data['OPENCV']['remake_3d'][:, 0], cam_data['OPENCV']['remake_3d'][:, 1],
                             cam_data['OPENCV']['remake_3d'][:, 2], color='green', alpha=0.5, marker='o', s=10,
                             label='OPENCV')
        handles1.append(handle)

        handle = ax2.scatter(led_index, np.array(cam_data['BLENDER']['distance_o']), color='red', alpha=0.5,
                             label='BLENDER distance_o')
        handles2.append(handle)

        handle = ax2.scatter(led_index, np.array(cam_data['BLENDER']['distance_t']), color='blue', alpha=0.5,
                             label='BLENDER distance_t')
        handles2.append(handle)

        handle = ax2.scatter(led_index, np.array(cam_data['OPENCV']['distance_o']), color='magenta', alpha=0.5,
                             label='OPENCV distance_o')
        handles2.append(handle)

        handle = ax2.scatter(led_index, np.array(cam_data['OPENCV']['distance_t']), color='green', alpha=0.5,
                             label='OPENCV distance_t')
        handles2.append(handle)

    # Create a dictionary to remove duplicates
    unique1 = {handle.get_label(): handle for handle in handles1}
    unique2 = {handle.get_label(): handle for handle in handles2}

    # Unpack the keys (labels) and values (handles) back into lists
    labels1, handles1 = zip(*unique1.items())
    labels2, handles2 = zip(*unique2.items())

    # Create legend
    ax1.legend(handles1, labels1)
    ax2.legend(handles2, labels2)

    plt.show()

def BA():
    pickle_file = 'bundle_adjustment.pickle'
    data = pickle_data(READ, pickle_file, None)
    CAMERA_INFO = data['CAMERA_INFO']
    camera_indices = []
    point_indices = []
    estimated_RTs = []
    POINTS_2D = []
    POINTS_3D = []

    LED_NUMBER = []
    n_points = 0
    cam_id = 0
    for key, camera_info in CAMERA_INFO.items():
        # print('key', key)
        # print(camera_info['points2D']['greysum'])
        # print(camera_info['led_num'])
        # print(camera_info['rt'])

        # Extract the camera id
        # cam_id = int(key.split('_')[-1])

        # Save camera parameters for each camera
        rvec = camera_info['BLENDER']['rt']['rvec']
        tvec = camera_info['BLENDER']['rt']['tvec']
        estimated_RTs.append((rvec.ravel(), tvec.ravel()))

        LED_NUMBER.append(camera_info['led_num'])
        # Save 3D and 2D points for each LED in the current camera
        for i in range(len(camera_info['led_num'])):
            # POINTS_3D.append(camera_info['points3D'][i])
            POINTS_3D.append(camera_info['BLENDER']['remake_3d'][i])
            POINTS_2D.append(camera_info['points2D']['greysum'][i])
            camera_indices.append(cam_id)
            point_indices.append(len(POINTS_3D) - 1)

        cam_id += 1
        # Add the number of 3D points in this camera to the total count
        n_points += len(camera_info['led_num'])

    def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))

        # set code function here
        points_proj = []
        for i, POINT_3D in enumerate(points_3d[point_indices]):
            camera_index = camera_indices[i]
            POINT_2D_PROJ, _ = cv2.projectPoints(POINT_3D,
                                                 np.array(camera_params[camera_indices][i][:3]),
                                                 np.array(camera_params[camera_indices][i][3:6]),
                                                 camera_matrix[CAM_ID][0],
                                                 camera_matrix[CAM_ID][1])
            points_proj.append(POINT_2D_PROJ[0][0])
            # print('points_3d', POINT_3D, ' ', camera_index, ' ', i)
            # print('R', np.array(camera_params[camera_indices][i][:3]))
            # print('T', np.array(camera_params[camera_indices][i][3:6]))

        points_proj = np.array(points_proj)
        return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
        m = camera_indices.size * 2
        n = n_cameras * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)
        i = np.arange(camera_indices.size)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1
        for s in range(3):
            A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
        return A

    # Convert the lists to NumPy arrays
    n_cameras = len(estimated_RTs)
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    camera_params = np.array(estimated_RTs)
    POINTS_2D = np.array(POINTS_2D).reshape(-1, 2)
    POINTS_3D = np.array(POINTS_3D).reshape(-1, 3)

    x0 = np.hstack((camera_params.ravel(), POINTS_3D.ravel()))
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    # print('n_cameras', n_cameras, 'n_points', n_points)
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, POINTS_2D))

    n_cam_params = res.x[:n_cameras * 6].reshape((n_cameras, 6))
    n_points_3d = res.x[n_cameras * 6:].reshape((n_points, 3))
    print("Optimized 3D points: ", n_points_3d, ' ', len(n_points_3d))
    print("Optimized camera parameters: ", n_cam_params)

    fig = plt.figure(figsize=(30, 10), num='Camera Simulator')
    plt.rcParams.update({'font.size': 7})
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    ax1 = plt.subplot(gs[0], projection='3d')
    # ax1 설정
    origin_pts = np.array(origin_led_data).reshape(-1, 3)
    ax1.set_title('3D plot')
    ax1.scatter(origin_pts[:, 0], origin_pts[:, 1], origin_pts[:, 2], color='black', alpha=0.5, marker='o', s=10)
    ax1.scatter(0, 0, 0, marker='o', color='k', s=20)
    ax1.set_xlim([-0.1, 0.1])
    ax1.set_xlabel('X')
    ax1.set_ylim([-0.1, 0.1])
    ax1.set_ylabel('Y')
    ax1.set_zlim([-0.1, 0.1])
    ax1.set_zlabel('Z')
    scale = 1.5
    f = zoom_factory(ax1, base_scale=scale)  # 이부분은 ax1이 zoom in, zoom out 기능을 가지게 해주는 코드입니다.
    ax1.scatter(n_points_3d[:, 0], n_points_3d[:, 1], n_points_3d[:, 2], color='blue', alpha=0.5, marker='o', s=3)

    plt.show()
def BA_3D_POINT():
    pickle_file = 'bundle_adjustment.pickle'
    data = pickle_data(READ, pickle_file, None)
    CAMERA_INFO = data['CAMERA_INFO']
    camera_indices = []
    point_indices = []
    estimated_RTs = []
    POINTS_2D = []
    POINTS_3D = []

    LED_NUMBER = []
    n_points = 0
    cam_id = 0
    for key, camera_info in CAMERA_INFO.items():
        # print('key', key)
        # print(camera_info['points2D']['greysum'])
        # print(camera_info['led_num'])
        # print(camera_info['rt'])

        # Extract the camera id
        # cam_id = int(key.split('_')[-1])

        # Save camera parameters for each camera
        rvec = camera_info['BLENDER']['rt']['rvec']
        tvec = camera_info['BLENDER']['rt']['tvec']
        estimated_RTs.append((rvec.ravel(), tvec.ravel()))

        LED_NUMBER.append(camera_info['led_num'])
        # Save 3D and 2D points for each LED in the current camera
        for i in range(len(camera_info['led_num'])):
            # POINTS_3D.append(camera_info['points3D'][i])
            POINTS_3D.append(camera_info['BLENDER']['remake_3d'][i])
            POINTS_2D.append(camera_info['points2D']['greysum'][i])
            camera_indices.append(cam_id)
            point_indices.append(len(POINTS_3D) - 1)

        cam_id += 1
        # Add the number of 3D points in this camera to the total count
        n_points += len(camera_info['led_num'])

    def fun(params, n_points, camera_indices, point_indices, points_2d, camera_params, camera_matrix):
        """Compute residuals.
        `params` contains 3-D coordinates.
        """
        points_3d = params.reshape((n_points, 3))

        points_proj = []
        for i, POINT_3D in enumerate(points_3d[point_indices]):
            camera_index = camera_indices[i]
            # print('points_3d', POINT_3D, ' ', camera_index, ' ', i)
            # print('R', np.array(camera_params[camera_indices][i][0]))
            # print('T', np.array(camera_params[camera_indices][i][1]))
            POINT_2D_PROJ, _ = cv2.projectPoints(POINT_3D,
                                                 np.array(camera_params[camera_indices][i][0]),
                                                 np.array(camera_params[camera_indices][i][1]),
                                                 camera_matrix[CAM_ID][0],
                                                 camera_matrix[CAM_ID][1])
            points_proj.append(POINT_2D_PROJ[0][0])

        points_proj = np.array(points_proj)
        return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity(n_points, point_indices):
        m = point_indices.size * 2
        n = n_points * 3
        A = lil_matrix((m, n), dtype=int)
        i = np.arange(point_indices.size)
        for s in range(3):
            A[2 * i, point_indices * 3 + s] = 1
            A[2 * i + 1, point_indices * 3 + s] = 1
        return A

    # Convert the lists to NumPy arrays
    n_cameras = len(estimated_RTs)
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    camera_params = np.array(estimated_RTs)
    POINTS_2D = np.array(POINTS_2D).reshape(-1, 2)
    POINTS_3D = np.array(POINTS_3D).reshape(-1, 3)

    # print('camera_params\n', camera_params.reshape(-1, 6))
    x0 = POINTS_3D.ravel()

    A = bundle_adjustment_sparsity(n_points, point_indices)
    print('n_points', n_points)
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_points, camera_indices, point_indices, POINTS_2D, camera_params, camera_matrix))

    # You are only optimizing points, so the result only contains point data
    n_points_3d = res.x.reshape((n_points, 3))
    print("Optimized 3D points: ", n_points_3d, ' ', len(n_points_3d))

    fig = plt.figure(figsize=(30, 10), num='Camera Simulator')
    plt.rcParams.update({'font.size': 7})
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    ax1 = plt.subplot(gs[0], projection='3d')
    # ax1 설정
    origin_pts = np.array(origin_led_data).reshape(-1, 3)
    ax1.set_title('3D plot')
    ax1.scatter(origin_pts[:, 0], origin_pts[:, 1], origin_pts[:, 2], color='black', alpha=0.5, marker='o', s=10)
    ax1.scatter(0, 0, 0, marker='o', color='k', s=20)
    ax1.set_xlim([-0.1, 0.1])
    ax1.set_xlabel('X')
    ax1.set_ylim([-0.1, 0.1])
    ax1.set_ylabel('Y')
    ax1.set_zlim([-0.1, 0.1])
    ax1.set_zlabel('Z')
    scale = 1.5
    f = zoom_factory(ax1, base_scale=scale)  # 이부분은 ax1이 zoom in, zoom out 기능을 가지게 해주는 코드입니다.
    ax1.scatter(n_points_3d[:, 0], n_points_3d[:, 1], n_points_3d[:, 2], color='blue', alpha=0.5, marker='o', s=3)

    plt.show()

if __name__ == "__main__":
    print(os.getcwd())
    print('origin')
    for i, leds in enumerate(origin_led_data):
        print(f"{i}, {leds}")
    print('target')
    for i, leds in enumerate(target_led_data):
        print(f"{i}, {leds}")

    gathering_data()
    # BA()
    # BA_3D_POINT()
