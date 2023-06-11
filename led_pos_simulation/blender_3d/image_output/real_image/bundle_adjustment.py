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
# Set the seed for Python's random module.
random.seed(1)
# Set the seed for NumPy's random module.
np.random.seed(1)
noise_std_dev = 0.001  # Noise standard deviation. Adjust this value to your needs.
# Generate noise with the same shape as the original data.
noise = np.random.normal(scale=noise_std_dev, size=origin_led_data.shape)
# Add noise to the original data.
noisy_led_data = origin_led_data + noise

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
    'points3D_N': [],
    'rt': {'rvec': [], 'tvec': []},
    'remake_3d': [],
    'distance': []
}
path = './bundle_area.json'
# Read the video
cap_0 = cv2.VideoCapture('./CAMERA_0_blender_test.mkv')


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
def gathering_data():
    # Read the first frame
    ret_0, frame_0 = cap_0.read()
    if not ret_0:
        print("Cannot read the video")
        cap_0.release()
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
            cap_0.release()
            return ERROR

        draw_blobs_and_ids(draw_frame, blob_area_0, trackers)
        cv2.imshow('image', draw_frame)
    cv2.destroyAllWindows()

    frame_cnt = 0
    while True:
        # Read a new frame
        ret_0, frame_0 = cap_0.read()
        if not ret_0:
            break

        _, frame_0 = cv2.threshold(cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                   cv2.THRESH_TOZERO)
        draw_frame = frame_0.copy()
        # Update trackers
        tracking_failed = []
        LED_NUMBER = []
        CAMERA_INFO[f"{frame_cnt}"] = copy.deepcopy(CAMERA_INFO_STRUCTURE)

        for id, data in trackers.items():
            ok, bbox = data['tracker'].update(frame_0)
            IDX = int(id)
            # Draw bounding box
            if ok:
                (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
                cv2.putText(draw_frame, f'{IDX}',  (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
                cx, cy = find_center(frame_0, (x, y, w, h))
                LED_NUMBER.append(IDX)
                CAMERA_INFO[f"{frame_cnt}"]['points2D']['greysum'].append([cx, cy])
                CAMERA_INFO[f"{frame_cnt}"]['points3D'].append(origin_led_data[IDX])
                CAMERA_INFO[f"{frame_cnt}"]['points3D_N'].append(noisy_led_data[IDX])
            else:
                if id not in tracking_failed:
                    tracking_failed.append(id)
                cv2.putText(draw_frame, f"Tracking failure detected {', '.join(map(str, tracking_failed))}", (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
        # Display result
        cv2.imshow("Tracking", draw_frame)

        if len(LED_NUMBER) >= 5:
            METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC
        else:
            METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P

        CAMERA_INFO[f"{frame_cnt}"]['led_num'] = LED_NUMBER
        points2D = np.array(CAMERA_INFO[f"{frame_cnt}"]['points2D']['greysum'], dtype=np.float64)
        points3D = np.array(CAMERA_INFO[f"{frame_cnt}"]['points3D'], dtype=np.float64)

        INPUT_ARRAY = [
            CAM_ID,
            points3D,
            points2D,
            camera_matrix[CAM_ID][0],
            camera_matrix[CAM_ID][1]
        ]
        ret, rvec, tvec, inliers = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
        CAMERA_INFO[f"{frame_cnt}"]['rt']['rvec'] = rvec
        CAMERA_INFO[f"{frame_cnt}"]['rt']['tvec'] = tvec

        # print('camera_info\n', CAMERA_INFO[f"{frame_cnt}"])

        frame_cnt += 1
        key = cv2.waitKey(1)
        # Exit if ESC key is
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == 27:
            print('ESC pressed')
            cap_0.release()
            cv2.destroyAllWindows()
            return ERROR

    cap_0.release()
    cv2.destroyAllWindows()
    file = 'bundle_adjustment.pickle'
    data = OrderedDict()
    data['CAMERA_INFO'] = CAMERA_INFO
    ret = pickle_data(WRITE, file, data)
    if ret != ERROR:
        print('data saved')
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
        rvec = camera_info['rt']['rvec']
        tvec = camera_info['rt']['tvec']
        estimated_RTs.append((rvec.ravel(), tvec.ravel()))

        LED_NUMBER.append(camera_info['led_num'])
        # Save 3D and 2D points for each LED in the current camera
        for i in range(len(camera_info['led_num'])):
            POINTS_3D.append(camera_info['points3D_N'][i])
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
        return (abs(points_proj - points_2d)).ravel()

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
        rvec = camera_info['rt']['rvec']
        tvec = camera_info['rt']['tvec']
        estimated_RTs.append((rvec.ravel(), tvec.ravel()))

        LED_NUMBER.append(camera_info['led_num'])
        # Save 3D and 2D points for each LED in the current camera
        for i in range(len(camera_info['led_num'])):
            POINTS_3D.append(camera_info['points3D_N'][i])
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
        return (abs(points_proj - points_2d)).ravel()

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
    print('noise')
    for i, leds in enumerate(noisy_led_data):
        print(f"{i}, {leds}")

    # gathering_data()
    # BA()
    BA_3D_POINT()
