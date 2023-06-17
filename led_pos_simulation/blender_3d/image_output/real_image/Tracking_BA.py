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
import os
import cv2
import glob
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
trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
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
        return 0, 0, 0

    return g_c_x, g_c_y, m_count
def detect_led_lights(image, padding=5, min_area=100, max_area=1000):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blob_info = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # Apply padding for the bounding box
        x -= padding
        y -= padding
        w += padding * 2
        h += padding * 2

        # Use contourArea to get the actual area of the contour
        area = cv2.contourArea(contour)

        # Check if the area of the contour is within the specified range
        if min_area <= area <= max_area:
            blob_info.append((x, y, w, h))

    return blob_info
def point_in_bbox(x, y, bbox):
    return bbox[0] <= x <= bbox[0] + bbox[2] and bbox[1] <= y <= bbox[1] + bbox[3]
def draw_blobs_and_ids(frame, blobs, bboxes):
    for bbox in blobs:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 1, 1)
    for box in bboxes:
        cv2.putText(frame, f"{box['idx']}", (int(box['bbox'][0]), int(box['bbox'][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
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
def click_event(event, x, y, flags, param, frame_0, blob_area_0, bboxes):
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, bbox in enumerate(blob_area_0):
            if point_in_bbox(x, y, bbox):
                input_number = input('Please enter ID for this bbox: ')
                bboxes.append({'idx': input_number, 'bbox': bbox})
                draw_blobs_and_ids(frame_0, blob_area_0, bboxes)
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
def mapping_id_blob(blob_centers, tcx, tcy):
    # Find blobs at the same position as the tracker
    same_position_blob = [
        (idx, math.sqrt((blob_centers[idx][0] - tcx) ** 2 + (blob_centers[idx][1] - tcy) ** 2)) for
        idx in range(len(blob_centers))]
    same_position_blob.sort(key=lambda x: x[1])
    if same_position_blob:
        # if the distance is within a certain threshold
        # Find blobs to the right of the tracker and sort them by their distances from the tracker
        led_candidates = [(idx,
                           blob_centers[idx][0],
                           blob_centers[idx][1],  # adding y coordinate
                           math.sqrt((blob_centers[idx][0] - tcx) ** 2 + (blob_centers[idx][1] - tcy) ** 2))
                          for idx in range(len(blob_centers)) if blob_centers[idx][0]]
        led_candidates.sort(key=lambda x: x[3])  # remember to change the sort key as well

        return led_candidates
def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.legacy.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker
def view_camera_infos(frame, text, x, y):
    cv2.putText(frame, text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), lineType=cv2.LINE_AA)
def blob_setting(script_dir):
    bboxes = []
    json_file = os.path.join(script_dir, './blob_area.json')
    json_data = rw_json_data(READ, json_file, None)
    if json_data != ERROR:
        bboxes = json_data['bboxes']
    image_files = sorted(glob.glob(os.path.join(script_dir, './tmp/render/*.png')))
    frame_0 = cv2.imread(image_files[0])
    if frame_0 is None:
        print("Cannot read the first image")
        cv2.destroyAllWindows()
        exit()

    _, frame_0 = cv2.threshold(cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                               cv2.THRESH_TOZERO)
    draw_frame = frame_0.copy()
    blob_area = detect_led_lights(draw_frame, 5, 5, 500)
    cv2.namedWindow('image')
    partial_click_event = functools.partial(click_event, frame_0=frame_0, blob_area_0=blob_area, bboxes=bboxes)
    cv2.setMouseCallback('image', partial_click_event)

    while True:
        key = cv2.waitKey(1)
        if key == ord('c'):
            print('clear area')
            bboxes.clear()
        elif key == ord('s'):
            print('save blob area')
            json_data = OrderedDict()
            json_data['bboxes'] = bboxes
            # Write json data
            rw_json_data(WRITE, json_file, json_data)

        elif key & 0xFF == 27:
            print('ESC pressed')
            cv2.destroyAllWindows()
            return

        elif key == ord('q'):
            print('go next step')
            break

        draw_blobs_and_ids(draw_frame, blob_area, bboxes)
        cv2.imshow('image', draw_frame)
    cv2.destroyAllWindows()

    return bboxes
def gathering_data_multi(bboxes):
    camera_params = read_camera_log('./tmp/render/camera_log.txt')
    image_files = sorted(glob.glob('./tmp/render/*.png'))
    BLOB_SIZE = 100
    frame_cnt = 0

    trackerType = "CSRT"
    # Create MultiTracker object
    multiTracker = cv2.legacy.MultiTracker_create()

    tracker_start = 0

    while frame_cnt < len(image_files):
        frame_0 = cv2.imread(image_files[frame_cnt])
        if frame_0 is None or frame_0.size == 0:
            print(f"Failed to load {image_files[frame_cnt]}, frame_cnt:{frame_cnt}")
            continue

        draw_frame = frame_0.copy()
        _, frame_0 = cv2.threshold(cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                   cv2.THRESH_TOZERO)
        cv2.putText(draw_frame, os.path.basename(image_files[frame_cnt]), (draw_frame.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        blob_area = detect_led_lights(frame_0, 5, 5, 500)
        blob_centers = []

        # Find Blob Area with findContours
        for blob_id, bbox in enumerate(blob_area):
            (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            gcx, gcy, gsize = find_center(frame_0, (x, y, w, h))
            # cv2.putText(draw_frame, f"{int(gcx)},{int(gcy)},{gsize}", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            if gsize < BLOB_SIZE:
                continue
            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
            blob_centers.append((gcx, gcy, bbox))


        # Initialize MultiTracker
        if tracker_start == 0:
            for i, data in enumerate(bboxes):
                multiTracker.add(createTrackerByName(trackerType), frame_0, data['bbox'])

        tracker_start = 1
        # get updated location of objects in subsequent frames
        qq, boxes = multiTracker.update(frame_0)

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(draw_frame, p1, p2, (0, 255, 0), 1)
            IDX = bboxes[i]['idx']
            view_camera_infos(draw_frame, ''.join([f'{IDX}']), int(newbox[0]), int(newbox[1]) - 10)

        frame_cnt += 1

        cv2.imshow("Tracking", draw_frame)
        key = cv2.waitKey(16)
        # Exit if ESC key is
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == 27:
            print('ESC pressed')
            cv2.destroyAllWindows()
            return ERROR
        elif key & 0xFF == ord('c'):
            while True:
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
def gathering_data_single(script_dir, bboxes):
    camera_params = read_camera_log(os.path.join(script_dir, './tmp/render/camera_log.txt'))
    image_files = sorted(glob.glob(os.path.join(script_dir, './tmp/render/*.png')))
    BLOB_SIZE = 50
    THRESHOLD_DISTANCE = 5
    frame_cnt = 0
    CURR_TRACKER = {}
    PREV_TRACKER = {}
    print('bboxes:', bboxes)
    if bboxes is None:
        return
    CURR_TRACKER[bboxes[0]['idx']] = {'bbox': bboxes[0]['bbox'], 'tracker': None}
    TRACKING_START = NOT_SET

    while frame_cnt < len(image_files):
        frame_0 = cv2.imread(image_files[frame_cnt])
        if frame_0 is None or frame_0.size == 0:
            print(f"Failed to load {image_files[frame_cnt]}, frame_cnt:{frame_cnt}")
            continue

        draw_frame = frame_0.copy()
        _, frame_0 = cv2.threshold(cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                   cv2.THRESH_TOZERO)
        cv2.putText(draw_frame, os.path.basename(image_files[frame_cnt]), (draw_frame.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        blob_area = detect_led_lights(frame_0, 5, 5, 500)
        if TRACKING_START == NOT_SET:
            init_trackers(CURR_TRACKER, frame_0)
        TRACKING_START = DONE

        CURR_TRACKER_CPY = CURR_TRACKER.copy()
        blob_centers = []
        print('CURR_TRACKER_CPY', CURR_TRACKER_CPY)

        # Find Blob Area with findContours
        for blob_id, bbox in enumerate(blob_area):
            (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            gcx, gcy, gsize = find_center(frame_0, (x, y, w, h))
            if gsize < BLOB_SIZE:
                continue
            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
            cv2.putText(draw_frame, f"{len(blob_centers)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            blob_centers.append((gcx, gcy, bbox))


        if len(CURR_TRACKER_CPY) > 0:
            Tracking_ANCHOR, Tracking_DATA = list(CURR_TRACKER_CPY.items())[0]
            ret, (tx, ty, tw, th) = Tracking_DATA['tracker'].update(frame_0)
            if ret:
                cv2.rectangle(draw_frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 1, 1)
                cv2.putText(draw_frame, f'{Tracking_ANCHOR}', (tx, ty + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                tcx, tcy, tsize = find_center(frame_0, (tx, ty, tw, th))
                # cv2.putText(draw_frame, f"{int(tcx)},{int(tcy)},{tsize}", (tx, ty + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

                if Tracking_ANCHOR in PREV_TRACKER:
                    dx = PREV_TRACKER[Tracking_ANCHOR][0] - tcx
                    dy = PREV_TRACKER[Tracking_ANCHOR][1] - tcy
                    euclidean_distance = math.sqrt(dx ** 2 + dy ** 2)
                    if euclidean_distance > THRESHOLD_DISTANCE or tsize < BLOB_SIZE:
                        print('euclidean_distance:', euclidean_distance, ' tsize:', tsize)
                        print('CUR_txy:', tcx, tcy)
                        print('PRV_txy:', PREV_TRACKER[Tracking_ANCHOR])
                        curr_pos_candidates = mapping_id_blob(blob_centers, tcx, tcy)
                        prev_pos_candidates = mapping_id_blob(blob_centers, PREV_TRACKER[Tracking_ANCHOR][0], PREV_TRACKER[Tracking_ANCHOR][1])
                        print('curr_pos_candidates:', curr_pos_candidates[0])
                        print('prev_pos_candidates:', prev_pos_candidates[0])
                        for bid, data in enumerate(blob_centers):
                            print(bid, ':', data)

                        del CURR_TRACKER[Tracking_ANCHOR]
                        del PREV_TRACKER[Tracking_ANCHOR]
                        print(f"tracker[{Tracking_ANCHOR}] deleted")

                        cv2.imshow("Tracking", draw_frame)
                        while True:
                            key = cv2.waitKey(1)
                            if key & 0xFF == ord('q'):
                                break
                        continue

                led_candidates = mapping_id_blob(blob_centers, tcx, tcy)
                if led_candidates != ERROR:
                    for i in range(min(3, len(led_candidates))):
                        if int(Tracking_ANCHOR) - i - 1 >= 0:
                            NEW_BLOB_ID = int(Tracking_ANCHOR) - i - 1
                            (cx, cy, cw, ch) = blob_centers[led_candidates[i][0]][2]
                            cv2.rectangle(draw_frame, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 1, 1)
                            cv2.putText(draw_frame, f'{NEW_BLOB_ID}', (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 255), 1)
                PREV_TRACKER[Tracking_ANCHOR] = (tcx, tcy)

            else:
                print('tracking failed')
                break

        frame_cnt += 1

        cv2.imshow("Tracking", draw_frame)
        key = cv2.waitKey(16)
        # Exit if ESC key is
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == 27:
            print('ESC pressed')
            cv2.destroyAllWindows()
            return ERROR
        elif key & 0xFF == ord('c'):
            while True:
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(os.getcwd())
    print('origin')
    for i, leds in enumerate(origin_led_data):
        print(f"{i}, {leds}")
    print('target')
    for i, leds in enumerate(target_led_data):
        print(f"{i}, {leds}")

    # trackers = init_blob_area()
    
    bboxes = blob_setting(script_dir)
    gathering_data_single(script_dir, bboxes)
