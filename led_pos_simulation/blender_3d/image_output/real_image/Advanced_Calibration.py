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
import sys
import bisect
import threading
import pprint
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from data_class import *
from itertools import combinations, permutations

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Add the directory containing poselib to the module search path
print(script_dir)
sys.path.append(os.path.join(script_dir, '../../../../EXTERNALS'))
# poselib only working in LINUX or WSL (window )
import poselib

RIFTS_PATTERN = [1,1,0,1,0,1,0,1,0,1,0,1,0,1,1]
ARCTURAS_PATTERN = [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
LEDS_POSITION = RIFTS_PATTERN

READ = 0
WRITE = 1
SUCCESS = 0
ERROR = -1
DONE = 1
NOT_SET = -1
CAM_ID = 0

AUTO_LOOP = 0
undistort = 1
THRESHOLD_DISTANCE = 10
TRACKING_ANCHOR_RECOGNIZE_SIZE = 1
max_level = 3
SHOW_PLOT = 1

FULL_COMBINATION_SEARCH = 0
DO_CALIBRATION_TEST = 1

CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
CV_MIN_THRESHOLD = 150
CV_MAX_THRESHOLD = 255
TRACKER_PADDING = 2
HOPPING_CNT = 4
BLOB_CNT = -1
# camera_log_path = "./tmp/render/ARCTURAS/camera_log.txt"
# camera_img_path = "./tmp/render/ARCTURAS/"
# BLOB_SIZE = 60
controller_name = 'rifts_left_2'
camera_log_path = f"./tmp/render/RIFTS/{controller_name}/camera_log.txt"
camera_img_path = f"./tmp/render/RIFTS/{controller_name}/"
BLOB_SIZE = 45

VIDEO_MODE = 0
video_img_path = 'output.mkv'
#Rift_S
calibrated_led_data_PCA = np.array([
[-0.0196017, -0.00410068, -0.0135735],
[-0.03277113, 0.00628887, -0.0135832],
[-0.03490691, 0.00833338, 0.00426659],
[-0.04303106, 0.0280018, -0.00324782],
[-0.04180463, 0.03545028, 0.02165716],
[-0.03023752, 0.06136769, 0.01576647],
[-0.01427574, 0.06341851, 0.03570797],
[0.00877676, 0.07193878, 0.02118695],
[0.03089874, 0.05445536, 0.03078265],
[0.03621063, 0.05241772, 0.01139059],
[0.04202049, 0.02990856, 0.01540787],
[0.04120085, 0.02222786, -0.00400265],
[0.03191536, 0.00400903, 0.00200633],
[0.03058087, 0.00350974, -0.01371021],
[0.01917571, -0.00225342, -0.01462277],
])
calibrated_led_data_IQR = np.array([
[-0.0196558, -0.00410544, -0.01356836],
[-0.03282789, 0.0062879, -0.01357887],
[-0.03493835, 0.0083316, 0.00427421],
[-0.0430377, 0.02800506, -0.00324572],
[-0.04180636, 0.03545349, 0.02166163],
[-0.03022426, 0.06137344, 0.01575953],
[-0.01429343, 0.06341453, 0.03570926],
[0.0087845, 0.07193409, 0.0211762],
[0.03090401, 0.0544413, 0.03077811],
[0.0361974, 0.05240306, 0.01137662],
[0.04201419, 0.02988591, 0.01540358],
[0.04120136, 0.02222655, -0.00399953],
[0.03192675, 0.00400135, 0.00201212],
[0.03072938, 0.00358713, -0.01370051],
[0.01917692, -0.0022665, -0.01462586],
])

# Arcturas
# calibrated_led_data_PCA = np.array([
# [-0.00331481, 0.03597334, 0.0041562],
# [0.00895387, 0.0480334, 0.00224952],
# [0.03186545, 0.05034012, 0.00452086],
# [0.05143521, 0.04647385, 0.00167996],
# [0.0691143, 0.02880501, 0.00437797],
# [0.07609677, 0.01258033, 0.00212461],
# [0.07808255, -0.00880987, 0.00175163],
# [0.07257527, -0.02523931, 0.00334523],
# [0.05415833, -0.04513621, 0.00300021],
# [0.03380799, -0.05154195, 0.00403185],
# [0.01199563, -0.04928216, 0.00324531],
# [-0.00577686, -0.03708742, 0.00419776],
# [-0.01801006, 0.02495871, 0.01861654],
# [-0.00627987, 0.03607813, 0.01639641],
# [0.02400921, 0.05223235, 0.01819005],
# [0.0439588, 0.04702286, 0.02071782],
# [0.06125609, 0.03599675, 0.0192615],
# [0.07299287, 0.01326892, 0.01876102],
# [0.07108337, -0.0200403, 0.01969068],
# [0.05584525, -0.04160456, 0.0191822],
# [0.03372417, -0.04916243, 0.01823499],
# [0.01221453, -0.04822629, 0.0171196],
# [-0.00584511, -0.03477198, 0.01903182],
# [-0.01835547, -0.02618116, 0.01613708],
# ])
# calibrated_led_data_IQR = np.array([
# [-0.00329021, 0.03592558, 0.00415626],
# [0.00896412, 0.04798009, 0.00222952],
# [0.03185886, 0.05032494, 0.00451729],
# [0.05141244, 0.04649456, 0.00167461],
# [0.06908566, 0.02877851, 0.00438188],
# [0.0760617, 0.01256322, 0.00213402],
# [0.07808777, -0.00880047, 0.00175405],
# [0.07264068, -0.02521716, 0.00334639],
# [0.05421395, -0.04515157, 0.00299299],
# [0.03368931, -0.05151773, 0.00403075],
# [0.01189147, -0.0491093, 0.00323798],
# [-0.00573937, -0.03708571, 0.00420315],
# [-0.01804127, 0.02493463, 0.01861341],
# [-0.00629089, 0.03607499, 0.01640096],
# [0.02402278, 0.05222171, 0.01818928],
# [0.04398363, 0.04701299, 0.02072038],
# [0.06127724, 0.0359907, 0.01926452],
# [0.07303191, 0.01326953, 0.01876604],
# [0.07112739, -0.02000828, 0.01969362],
# [0.05574092, -0.04158745, 0.01918677],
# [0.03372719, -0.049154, 0.01824005],
# [0.01226819, -0.04825089, 0.0171238],
# [-0.00582242, -0.03479266, 0.01903632],
# [-0.01831358, -0.02621611, 0.01612675],
# ])

camera_matrix = [
    [np.array([[712.623, 0.0, 653.448],
               [0.0, 712.623, 475.572],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)],
]
default_dist_coeffs = np.zeros((4, 1))
default_cameraK = np.eye(3).astype(np.float64)
blob_status = [[0, 'new'] for _ in range(BLOB_CNT)]
CAMERA_INFO = {}
CAMERA_INFO_STRUCTURE = {
    'LED_NUMBER': [],
    'points2D': {'greysum': [], 'opencv': [], 'blender': []},
    'points2D_U': {'greysum': [], 'opencv': [], 'blender': []},
    'points3D': [],
    'points3D_PCA': [],
    'points3D_IQR': [],
    'BLENDER': {'rt': {'rvec': [], 'tvec': []}},
    'OPENCV': {'rt': {'rvec': [], 'tvec': []}},
}
BLOB_INFO = {}
BLOB_INFO_STRUCTURE = {
    'points2D_D': {'greysum': []},
    'points2D_U': {'greysum': []},
    'BLENDER': {'rt': {'rvec': [], 'tvec': []}},
    'OPENCV': {'rt': {'rvec': [], 'tvec': []}},
}

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
            if frame[y][x] >= CV_MIN_THRESHOLD:
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
        cv2.putText(frame, f"{box['idx']}", (int(box['bbox'][0]), int(box['bbox'][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 1)
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
        (x, y, w, h) = data['bbox']        
        ok = tracker.init(frame, (x - TRACKER_PADDING, y - TRACKER_PADDING, w + 2 * TRACKER_PADDING, h + 2 * TRACKER_PADDING))
        data['tracker'] = tracker
def click_event(event, x, y, flags, param, frame, blob_area_0, bboxes):
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, bbox in enumerate(blob_area_0):
            if point_in_bbox(x, y, bbox):
                input_number = input('Please enter ID for this bbox: ')
                bboxes.append({'idx': input_number, 'bbox': bbox})
                draw_blobs_and_ids(frame, blob_area_0, bboxes)
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
def mapping_id_blob(blob_centers, Tracking_ANCHOR, TRACKER):
    tcx = TRACKER[Tracking_ANCHOR][0]
    tcy = TRACKER[Tracking_ANCHOR][1]
    Tracking_ANCHOR = int(Tracking_ANCHOR)
    # Calculate distances for all blobs
    blob_distances = [(idx,
                       blob_centers[idx][0],
                       blob_centers[idx][1],  
                       math.sqrt((blob_centers[idx][0] - tcx) ** 2 + (blob_centers[idx][1] - tcy) ** 2),
					   -1)
                      for idx in range(len(blob_centers))]

    # Get blobs to the left of the tracker and sort them
    led_candidates_left = sorted(
        (blob for blob in blob_distances if blob[1] <= tcx),
        key=lambda blob: (-blob[1], blob[3])
    )
    # For blobs that are very close in x, prioritize the one closer to the tracker in y
    for i in range(len(led_candidates_left) - 1):
        if abs(led_candidates_left[i][1] - led_candidates_left[i + 1][1]) == 0:
            if led_candidates_left[i][3] > led_candidates_left[i + 1][3]:
                led_candidates_left[i], led_candidates_left[i + 1] = led_candidates_left[i + 1], led_candidates_left[i]
    # Do the same for blobs to the right of the tracker
    led_candidates_right = sorted(
        (blob for blob in blob_distances if blob[1] >= tcx),
        key=lambda blob: (blob[1], blob[3])
    )
    for i in range(len(led_candidates_right) - 1):
        if abs(led_candidates_right[i][1] - led_candidates_right[i + 1][1]) == 0:
            if led_candidates_right[i][3] > led_candidates_right[i + 1][3]:
                led_candidates_right[i], led_candidates_right[i + 1] = led_candidates_right[i + 1], led_candidates_right[i]
    ANCHOR_POS = LEDS_POSITION[Tracking_ANCHOR]
    clockwise = 0
    counterclockwise = 1
    TOP = 0
    BOTTOM = 1
    def BLOB_ID_SEARCH(status, position, direction):
        # print(f"{status} {position} {direction}")
        COUNT = 1 
        BLOB_ID = -1
        while True:
            if direction == clockwise:
                POSITION_SEARCH = position + COUNT
                POSITION_SEARCH %= len(LEDS_POSITION)
                temp_id = LEDS_POSITION[POSITION_SEARCH]
                if status == TOP:                    
                    if temp_id != TOP:
                        COUNT += 1
                    else:
                        BLOB_ID = POSITION_SEARCH
                        break
                elif status == BOTTOM:
                    if temp_id != BOTTOM:
                        COUNT += 1
                    else:
                        BLOB_ID = POSITION_SEARCH
                        break          
            elif direction == counterclockwise:
                POSITION_SEARCH = position - COUNT
                if POSITION_SEARCH < 0:
                    POSITION_SEARCH += len(LEDS_POSITION)
                
                temp_id = LEDS_POSITION[POSITION_SEARCH]
                if status == TOP:                    
                    if temp_id != TOP:
                        COUNT += 1
                    else:
                        BLOB_ID = POSITION_SEARCH
                        break
                elif status == BOTTOM:
                    if temp_id != BOTTOM:
                        COUNT += 1
                    else:
                        BLOB_ID = POSITION_SEARCH
                        break
        return BLOB_ID                
    # Insert Blob ID here
    # print('BEFORE')
    # print('left_data')
    LEFT_BLOB_INFO = [-1, -1]
    led_candidates_left = [list(t) for t in led_candidates_left]
    for left_data in led_candidates_left:
        # print(left_data)
        if left_data[3] < TRACKING_ANCHOR_RECOGNIZE_SIZE:
            left_data[4] = Tracking_ANCHOR
            continue
        # TOP Searching and clockwise
        if ANCHOR_POS == TOP:
            if left_data[2] <= CAP_PROP_FRAME_HEIGHT / 2:
                CURR_ID = Tracking_ANCHOR if LEFT_BLOB_INFO[TOP] == -1 else LEFT_BLOB_INFO[TOP]
                NEW_BLOB_ID = BLOB_ID_SEARCH(TOP, CURR_ID, clockwise)
                left_data[4] = NEW_BLOB_ID
                LEFT_BLOB_INFO[TOP] = NEW_BLOB_ID
        else:
            # BOTTOM Searching and clockwise
            if left_data[2] > CAP_PROP_FRAME_HEIGHT / 2:
                CURR_ID = Tracking_ANCHOR if LEFT_BLOB_INFO[BOTTOM] == -1 else LEFT_BLOB_INFO[BOTTOM]
                NEW_BLOB_ID = BLOB_ID_SEARCH(BOTTOM, CURR_ID, clockwise)
                left_data[4] = NEW_BLOB_ID
                LEFT_BLOB_INFO[BOTTOM] = NEW_BLOB_ID
        
    # print('right_data')
    led_candidates_right = [list(t) for t in led_candidates_right]
    RIGHT_BLOB_INFO = [-1, -1]
    for right_data in led_candidates_right:
        # print(right_data)
        if right_data[3] < TRACKING_ANCHOR_RECOGNIZE_SIZE:
            right_data[4] = Tracking_ANCHOR
            continue

        if ANCHOR_POS == TOP:
            if right_data[2] <= CAP_PROP_FRAME_HEIGHT / 2:
                CURR_ID = Tracking_ANCHOR if RIGHT_BLOB_INFO[TOP] == -1 else RIGHT_BLOB_INFO[TOP]
                NEW_BLOB_ID = BLOB_ID_SEARCH(TOP, CURR_ID, counterclockwise)
                right_data[4] = NEW_BLOB_ID
                RIGHT_BLOB_INFO[TOP] = copy.deepcopy(NEW_BLOB_ID)
        else:
            if right_data[2] > CAP_PROP_FRAME_HEIGHT / 2:
                CURR_ID = Tracking_ANCHOR if RIGHT_BLOB_INFO[BOTTOM] == -1 else RIGHT_BLOB_INFO[BOTTOM]
                NEW_BLOB_ID = BLOB_ID_SEARCH(BOTTOM, CURR_ID, counterclockwise)
                right_data[4] = NEW_BLOB_ID
                RIGHT_BLOB_INFO[BOTTOM] = copy.deepcopy(NEW_BLOB_ID)

    # Remove rows where the 4th value is -1 or the 3rd value is less than or equal to 1
    led_candidates_left = [row for row in led_candidates_left if row[4] != -1]
    led_candidates_right = [row for row in led_candidates_right if row[4] != -1]

    # print('AFTER')
    # print('left_data')
    # for left_data in led_candidates_left:
    #     print(left_data)
    # print('right_data')
    # for right_data in led_candidates_right:
    #     print(right_data)

    return led_candidates_left, led_candidates_right
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
    print('blob_setting START')
    bboxes = []
    json_file = os.path.join(script_dir, './blob_area.json')
    json_data = rw_json_data(READ, json_file, None)
    if json_data != ERROR:
        bboxes = json_data['bboxes']
    image_files = sorted(glob.glob(os.path.join(script_dir, camera_img_path + '*.png')))
    camera_params = read_camera_log(os.path.join(script_dir, camera_log_path))
    if VIDEO_MODE == 1:
        video = cv2.VideoCapture(video_img_path)
    frame_cnt = 0
    while video.isOpened() if VIDEO_MODE else True:
        if VIDEO_MODE == 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_cnt)  # Frame indices start from 0
            ret, frame = video.read()
            filename = f"VIDEO Mode {video_img_path}"
            if not ret:
                break
        else:
            if frame_cnt >= len(image_files):
                break
            frame = cv2.imread(image_files[frame_cnt])
            filename = f"IMAGE Mode {os.path.basename(image_files[frame_cnt])}"
            if frame is None:
                print("Cannot read the first image")
                cv2.destroyAllWindows()
                exit()

        _, frame = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                   cv2.THRESH_TOZERO)
        draw_frame = frame.copy()
        height, width = draw_frame.shape
        center_x, center_y = width // 2, height // 2
        cv2.line(draw_frame, (0, center_y), (width, center_y), (255, 0, 0), 1)
        cv2.line(draw_frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)
        brvec, btvec = camera_params[frame_cnt + 1]
        
        cv2.putText(draw_frame, f"frame_cnt {frame_cnt} [{filename}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.putText(draw_frame, f"rvec: {brvec}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(draw_frame, f"tvec: {btvec}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        blob_area = detect_led_lights(frame, 2, 5, 500)
        cv2.namedWindow('image')
        partial_click_event = functools.partial(click_event, frame=frame, blob_area_0=blob_area, bboxes=bboxes)
        cv2.setMouseCallback('image', partial_click_event)
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
        elif key == ord('n'):
            frame_cnt += 1
            bboxes.clear()
            print('go next IMAGE')
        elif key == ord('b'):
            frame_cnt -= 1
            bboxes.clear()
            print('go back IMAGE')

        draw_blobs_and_ids(draw_frame, blob_area, bboxes)
        cv2.imshow('image', draw_frame)

    cv2.destroyAllWindows()

    return bboxes
def quat_to_rotm(q):
    qw, qx, qy, qz = q
    qx2, qy2, qz2 = qx * qx, qy * qy, qz * qz
    qxqy, qxqz, qyqz = qx * qy, qx * qz, qy * qz
    qxqw, qyqw, qzqw = qx * qw, qy * qw, qz * qw

    return np.array([[1 - 2 * (qy2 + qz2), 2 * (qxqy - qzqw), 2 * (qxqz + qyqw)],
                     [2 * (qxqy + qzqw), 1 - 2 * (qx2 + qz2), 2 * (qyqz - qxqw)],
                     [2 * (qxqz - qyqw), 2 * (qyqz + qxqw), 1 - 2 * (qx2 + qy2)]])
def init_new_tracker(prev_frame, Tracking_ANCHOR, CURR_TRACKER, PREV_TRACKER):
    # prev_frame = cv2.imread(frame)
    if prev_frame is None or prev_frame.size == 0:
        print(f"Failed to load prev frame")
        return ERROR, None
    # draw_frame = prev_frame.copy()
    _, prev_frame = cv2.threshold(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                  cv2.THRESH_TOZERO)
    # filename = os.path.basename(frame)
    # cv2.putText(draw_frame, f"{filename}", (draw_frame.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 1)
    # find Blob area by findContours
    blob_area = detect_led_lights(prev_frame, 2, 5, 500)
    blob_centers = []
    for blob_id, bbox in enumerate(blob_area):
        (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        gcx, gcy, gsize = find_center(prev_frame, (x, y, w, h))
        if gsize < BLOB_SIZE:
            continue
        # cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
        # cv2.putText(draw_frame, f"{blob_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        blob_centers.append((gcx, gcy, bbox))
    prev_pos_candidates, _ = mapping_id_blob(blob_centers, Tracking_ANCHOR, PREV_TRACKER)
    print('prev_pos_candidates:', prev_pos_candidates[0])
    # for bid, data in enumerate(blob_centers):
    #     print(bid, ':', data)

    NEW_Tracking_ANCHOR = -1
    NEW_Tracking_bbox = None    

    for i in range(min(HOPPING_CNT, len(prev_pos_candidates))):
        NEW_BLOB_ID = prev_pos_candidates[i][4]
        NEW_Tracking_bbox = blob_centers[prev_pos_candidates[i][0]][2]
        NEW_Tracking_ANCHOR = NEW_BLOB_ID
    if NEW_Tracking_ANCHOR != -1 and NEW_Tracking_bbox is not None:
        CURR_TRACKER_CPY = CURR_TRACKER.copy()
        for Other_side_Tracking_ANCHOR, _ in CURR_TRACKER_CPY.items():
            if Other_side_Tracking_ANCHOR != Tracking_ANCHOR:
                Other_Tracking_bbox = PREV_TRACKER[Other_side_Tracking_ANCHOR][2]
                CURR_TRACKER[Other_side_Tracking_ANCHOR] = {'bbox': Other_Tracking_bbox, 'tracker': None}
                init_trackers(CURR_TRACKER, prev_frame)

        # print(f"UPDATE NEW_Tracking_ANCHOR {NEW_Tracking_ANCHOR} NEW_Tracking_bbox {NEW_Tracking_bbox}")
        CURR_TRACKER[NEW_Tracking_ANCHOR] = {'bbox': NEW_Tracking_bbox, 'tracker': None}
        init_trackers(CURR_TRACKER, prev_frame)
        # tcx, tcy, _ = find_center(prev_frame, NEW_Tracking_bbox)
        # PREV_TRACKER[NEW_Tracking_ANCHOR] = (tcx, tcy)
        # cv2.imshow('TRACKER change', draw_frame)
        # cv2.waitKey(0)
        
        return SUCCESS, CURR_TRACKER
    else:
        return ERROR, None
def is_valid_pose(pose):
    q, t = pose.q, pose.t
    return not np.isnan(q).any() and not np.isnan(t).any()
def crop_image(image, x, y, w, h):
    return image[y:y + h, x:x + w]
def save_image(image, name):
    cv2.imwrite(f"{name}.png", image)
def find_circles_or_ellipses(image, draw_frame):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    # Apply threshold
    ret, threshold_image = cv2.threshold(blurred_image, 150, 255, cv2.THRESH_BINARY)

    # Perform Edge detection on the thresholded image
    edges = cv2.Canny(threshold_image, 150, 255)

    padding = 2
    height, width = image.shape[:2]
    max_radius = int(min(width, height) / 2 - padding)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=50, minRadius=0,
                               maxRadius=max_radius)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circle_count = 0
    ellipse_count = 0

    if circles is not None:
        for circle in circles[0, :]:
            # Convert the center coordinates to int
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])
            cv2.circle(draw_frame, center, radius, (255, 0, 0), 1)
            circle_count += 1

    for contour in contours:
        if len(contour) >= 5:  # A contour must have at least 5 points for fitEllipse
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(draw_frame, ellipse, (0, 0, 255), 1)
            ellipse_count += 1

    # Draw all contours for debugging purposes
    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

    # Check if we found multiple blobs or not a circle/ellipse
    if circle_count + ellipse_count > 1:
        print(f"Detected {circle_count} circles and {ellipse_count} ellipses")
        return True, image, draw_frame

    return False, image, draw_frame
def check_blobs_with_pyramid(image, draw_frame, x, y, w, h, max_level):
    # Crop the image
    cropped = crop_image(image, x, y, w, h)

    # Generate image pyramid using pyrUp (increasing resolution)
    gaussian_pyramid = [cropped]
    draw_pyramid = [crop_image(draw_frame, x, y, w, h)]  # Also generate pyramid for draw_frame
    for i in range(max_level):
        cropped = cv2.pyrUp(cropped)
        draw_frame_cropped = cv2.pyrUp(draw_pyramid[-1])
        gaussian_pyramid.append(cropped)
        draw_pyramid.append(draw_frame_cropped)

    FOUND_STATUS = False
    # Check for circles or ellipses at each level
    for i, (img, draw_frame_cropped) in enumerate(zip(gaussian_pyramid, draw_pyramid)):
        found, img_with_contours, draw_frame_with_shapes = find_circles_or_ellipses(img.copy(),
                                                                                    draw_frame_cropped.copy())
        # Save the image for debugging
        if found:
            FOUND_STATUS = True
            # save_image(img_with_contours, f"debug_{i}_{FOUND_STATUS}")
            # save_image(draw_frame_with_shapes, f"debug_draw_{i}_{FOUND_STATUS}")
    return FOUND_STATUS
def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    if np.linalg.matrix_rank(H) < 3:
        raise ValueError("rank of H = {}, expecting 3".format(np.linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t
def module_lsm(blob_data):
    print(module_lsm.__name__)
    # X,Y,Z coordination
    axis_cnt = 3
    origin_pts = []
    before_pts = []
    after_pts = []
    led_array = []
    for led_id, points in enumerate(blob_data):
        led_array.append(led_id)
        origin_pts.append(MODEL_DATA[led_id])
        before_pts.append(points)

    origin_pts = np.array(origin_pts)
    before_pts = np.array(before_pts)
    led_blob_cnt = len(led_array)

    # make 3xN matrix
    A = np.array([[0 for j in range(led_blob_cnt)] for i in range(axis_cnt)], dtype=float)
    B = np.array([[0 for j in range(led_blob_cnt)] for i in range(axis_cnt)], dtype=float)
    for r in range(led_blob_cnt):
        for c in range(axis_cnt):
            B[c][r] = origin_pts[r][c]
            A[c][r] = before_pts[r][c]

    # calculation rigid_transform
    ret_R, ret_t = rigid_transform_3D(A, B)
    C = (ret_R @ A) + ret_t

    for id in (led_array):
        after_pts.append([float(C[0][id]), float(C[1][id]), float(C[2][id])])

    diff = np.array(C - B)
    err = C - B
    dist = []
    for i in range(len(diff[0])):
        dist.append(np.sqrt(np.power(diff[0][i], 2) + np.power(diff[1][i], 2) + np.power(diff[2][i], 2)))
    err = err * err
    err = np.sum(err)
    rmse = np.sqrt(err / len(diff[0]))

    # print('rmse')
    # print(rmse)
    # print('A')
    # print(A)
    # print('B')
    # print(B)
    # print('C')
    # print(C)
    # print('diff')
    # print(diff)
    # print('dist')
    # print(dist)

    return after_pts
def calculate_camera_position_direction(rvec, tvec):
    # Extract rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    R_inv = np.linalg.inv(R)
    # Camera position (X, Y, Z)
    Cam_pos = -R_inv @ tvec
    X, Y, Z = Cam_pos.ravel()

    unit_z = np.array([0, 0, 1])

    # idea 1
    # roll = math.atan2(R[2][1], R[2][2])
    # pitch = math.atan2(-R[2][0], math.sqrt(R[2][1]**2 + R[2][2]**2))
    # yaw = math.atan2(R[1][0], R[0][0])

    # idea 2
    Zc = np.reshape(unit_z, (3, 1))
    Zw = np.dot(R_inv, Zc)  # world coordinate of optical axis
    zw = Zw.ravel()

    pan = np.arctan2(zw[1], zw[0]) - np.pi / 2
    tilt = np.arctan2(zw[2], np.sqrt(zw[0] * zw[0] + zw[1] * zw[1]))

    # roll
    unit_x = np.array([1, 0, 0])
    Xc = np.reshape(unit_x, (3, 1))
    Xw = np.dot(R_inv, Xc)  # world coordinate of camera X axis
    xw = Xw.ravel()
    xpan = np.array([np.cos(pan), np.sin(pan), 0])

    roll = np.arccos(np.dot(xw, xpan))  # inner product
    if xw[2] < 0:
        roll = -roll

    roll = roll
    pitch = tilt
    yaw = pan

    optical_axis = R.T @ unit_z.T
    optical_axis_x, optical_axis_y, optical_axis_z = optical_axis

    return (X, Y, Z), (optical_axis_x, optical_axis_y, optical_axis_z), (roll, pitch, yaw)
def check_angle_and_facing(points3D, cam_pos, cam_dir, blob_ids, threshold_angle=80.0):
    results = {}
    # RVECS = np.array([[cam_dir[0]], [cam_dir[1]], [cam_dir[2]]], dtype=np.float64)
    # cam_ori = R.from_rotvec(RVECS.reshape(3)).as_quat()
    cam_pose = {'position': vector3(*cam_pos), 'orient': quat(*cam_dir)}
    for pts3D_idx, blob_id in enumerate(blob_ids):
        # Blob의 위치
        blob_pos = points3D[pts3D_idx]        
        # Blob의 방향 벡터
        blob_dir = DIRECTION[blob_id]
        # Blob의 위치와 방향 벡터를 카메라 pose에 맞게 변환
        temp = transfer_point(vector3(*blob_pos), cam_pose)
        ori = rotate_point(vector3(*blob_dir), cam_pose)
        # facing_dot 찾기
        facing_dot = get_dot_point(nomalize_point(temp), ori)
        angle_rad = math.radians(180.0 - threshold_angle)
        rad = np.cos(angle_rad)
        # threshold_angle 이내인지 확인하고, facing_dot가 rad보다 작지 않은지 확인
        results[blob_id] = (facing_dot < rad)
        print(f"{blob_id} : facing_dot: {facing_dot} rad{rad}")
    return results
def cal_iqr_func(arr):
    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)

    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    lower_bound = Q1 - outlier_step
    upper_bound = Q3 + outlier_step

    mask = np.where((arr > upper_bound) | (arr < lower_bound))
    return mask
def detect_outliers(blob_array, remove_index_array):
    for blob_data in blob_array:
        if len(blob_data) != 0:
            temp_x = np.array(cal_iqr_func(blob_data))
            for x in temp_x:
                for xx in x:
                    if xx in remove_index_array:
                        continue
                    else:
                        remove_index_array.append(xx)
    remove_index_array.sort()
def gathering_data_single(ax1, script_dir, bboxes):
    print('gathering_data_single START')
    camera_params = read_camera_log(os.path.join(script_dir, camera_log_path))
    image_files = sorted(glob.glob(os.path.join(script_dir, camera_img_path + '*.png')))

    CURR_TRACKER = {}
    PREV_TRACKER = {}

    print('bboxes:', bboxes)
    if bboxes is None:
        return
    
    for i in range(len(bboxes)):
        CURR_TRACKER[bboxes[i]['idx']] = {'bbox': bboxes[i]['bbox'], 'tracker': None}

    # Init Multi Tracker
    TRACKING_START = NOT_SET
    # trackerType = "CSRT"
    # multiTracker = cv2.legacy.MultiTracker_create()

    mutex = threading.Lock()
    # Initialize each blob ID with a copy of the structure
    for blob_id in range(BLOB_CNT):  # blob IDs: 0 to 14
        BLOB_INFO[blob_id] = copy.deepcopy(BLOB_INFO_STRUCTURE)

    if VIDEO_MODE == 1:
        video = cv2.VideoCapture(video_img_path)
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        print('lenght of images: ', total_frames)
    else:
        print('lenght of images: ', len(image_files))

    frame_cnt = 0
    while video.isOpened() if VIDEO_MODE else True:
        print('\n')
        print(f"########## Frame {frame_cnt} ##########")
        if VIDEO_MODE == 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_cnt)
            ret, frame_0 = video.read()
            filename = f"VIDEO Mode {video_img_path}"
            if not ret:
                break
        else:
            # BLENDER와 확인해 보니 마지막 카메라 위치가 시작지점으로 돌아와서 추후 remake 3D 에서 이상치 발생 ( -1 )  
            if frame_cnt >= len(image_files) - 1:
                break
            frame_0 = cv2.imread(image_files[frame_cnt])
            filename = f"IMAGE Mode {os.path.basename(image_files[frame_cnt])}"
            if frame_0 is None or frame_0.size == 0:
                print(f"Failed to load {image_files[frame_cnt]}, frame_cnt:{frame_cnt}")
                continue

        draw_frame = frame_0.copy()
        _, frame_0 = cv2.threshold(cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                   cv2.THRESH_TOZERO)

        cv2.putText(draw_frame, f"frame_cnt {frame_cnt} [{filename}]", (draw_frame.shape[1] - 300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        height, width = frame_0.shape
        center_x, center_y = width // 2, height // 2
        cv2.line(draw_frame, (0, center_y), (width, center_y), (255, 255, 255), 1)
        cv2.line(draw_frame, (center_x, 0), (center_x, height), (255, 255, 255), 1)        
        # print('CURR_TRACKER', CURR_TRACKER)

        if TRACKING_START == NOT_SET:
            init_trackers(CURR_TRACKER, frame_0)
            # for i, data in enumerate(bboxes):
            #     multiTracker.add(createTrackerByName(trackerType), frame_0, data['bbox'])
        TRACKING_START = DONE

        # find Blob area by findContours
        blob_area = detect_led_lights(frame_0, 2, 5, 500)
        blob_centers = []    
        for blob_id, bbox in enumerate(blob_area):
            (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            gcx, gcy, gsize = find_center(frame_0, (x, y, w, h))
            if gsize < BLOB_SIZE:
                continue

            overlapping = check_blobs_with_pyramid(frame_0, draw_frame, x, y, w, h, max_level)
            if overlapping == True:
                cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 0, 255), 1, 1)
                cv2.putText(draw_frame, f"SEG", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                continue

            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
            # cv2.putText(draw_frame, f"{blob_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # cv2.putText(draw_frame, f"{int(gcx)},{int(gcy)},{gsize}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
            #             (255, 255, 255), 1)
            blob_centers.append((gcx, gcy, bbox))
            # print(f"{blob_id} : {gcx}, {gcy}")

        CURR_TRACKER_CPY = CURR_TRACKER.copy()
        # print('CURR_TRACKER_CPY', CURR_TRACKER_CPY)

        if len(CURR_TRACKER_CPY) > 0 and frame_cnt + 1 < len(camera_params):
            brvec, btvec = camera_params[frame_cnt + 1]
            brvec_reshape = np.array(brvec).reshape(-1, 1)
            btvec_reshape = np.array(btvec).reshape(-1, 1)
            # print('Blender rvec:', brvec_reshape.flatten(), ' tvec:', btvec_reshape.flatten())

            TEMP_BLOBS = {}
            TRACKER_BROKEN_STATUS = NOT_SET
            for Tracking_ANCHOR, Tracking_DATA in CURR_TRACKER_CPY.items():
                if Tracking_DATA['tracker'] is not None:
                    print('Tracking_ANCHOR:', Tracking_ANCHOR)
                    ret, (tx, ty, tw, th) = Tracking_DATA['tracker'].update(frame_0)
                    # cv2.rectangle(draw_frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 1, 1)                    
                    cv2.putText(draw_frame, f'{Tracking_ANCHOR}', (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                    tcx, tcy, tsize = find_center(frame_0, (tx, ty, tw, th))
                    cv2.circle(draw_frame, (int(tcx), int(tcy)), 2, (0, 255, 0), -1)
                    if Tracking_ANCHOR in PREV_TRACKER:
                        def check_distance(blob_centers, tcx, tcy):
                            for center in blob_centers:
                                gcx, gcy, _ = center
                                distance = ((gcx - tcx)**2 + (gcy - tcy)**2)**0.5
                                if distance < TRACKING_ANCHOR_RECOGNIZE_SIZE:
                                    return False
                            return True
                        dx = PREV_TRACKER[Tracking_ANCHOR][0] - tcx
                        dy = PREV_TRACKER[Tracking_ANCHOR][1] - tcy
                        euclidean_distance = math.sqrt(dx ** 2 + dy ** 2)
                        # 트랙커가 갑자기 이동
                        # 사이즈가 작은 경우
                        # 실패한 경우
                        # 중심점위치에 Blob_center 데이터가 없는 경우
                        exist_status = check_distance(blob_centers, tcx, tcy)
                        if exist_status or euclidean_distance > THRESHOLD_DISTANCE or tsize < BLOB_SIZE or not ret:
                            print('Tracker Broken')
                            print('euclidean_distance:', euclidean_distance, ' tsize:', tsize, ' ret:', ret, 'exist_status:', exist_status)
                            print('CUR_txy:', tcx, tcy)
                            print('PRV_txy:', PREV_TRACKER[Tracking_ANCHOR])
                            # del CURR_TRACKER[Tracking_ANCHOR]
                            if VIDEO_MODE:
                                video.set(cv2.CAP_PROP_POS_FRAMES, frame_cnt - 1)
                                ret, search_frame = video.read()
                            else:
                                search_frame =  image_files[frame_cnt - 1]
                                search_frame = cv2.imread(search_frame)      
                            ret, CURR_TRACKER = init_new_tracker(search_frame, Tracking_ANCHOR, CURR_TRACKER, PREV_TRACKER)

                            if ret == SUCCESS:
                                del CURR_TRACKER[Tracking_ANCHOR]
                                del PREV_TRACKER[Tracking_ANCHOR]
                                # 여기서 PREV에 만들어진 위치를 집어넣어야 바로 안튕김
                                print(f"tracker[{Tracking_ANCHOR}] deleted")
                                frame_cnt -= 1
                                TRACKER_BROKEN_STATUS = DONE
                                break
                            else:
                                print('Tracker Change Failed')
                                cv2.imshow("Tracking", draw_frame)
                                while True:
                                    key = cv2.waitKey(1)
                                    if key & 0xFF == ord('q'):
                                        break
                                break

                    PREV_TRACKER[Tracking_ANCHOR] = (tcx, tcy, (tx, ty, tw, th))
                    led_candidates_left, led_candidates_right = mapping_id_blob(blob_centers, Tracking_ANCHOR, PREV_TRACKER) 

                    for i in range(len(led_candidates_left)):
                        NEW_BLOB_ID = led_candidates_left[i][4]
                        (cx, cy, cw, ch) = blob_centers[led_candidates_left[i][0]][2]                        
                        cv2.rectangle(draw_frame, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 1, 1)
                        if NEW_BLOB_ID != int(Tracking_ANCHOR):
                            cv2.putText(draw_frame, f'{NEW_BLOB_ID}', (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 255, 255), 1)
                        points2D_D = np.array([blob_centers[led_candidates_left[i][0]][0], blob_centers[led_candidates_left[i][0]][1]], dtype=np.float64)
                        points2D_U = np.array(cv2.undistortPoints(points2D_D, camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][1])).reshape(-1, 2)
                        TEMP_BLOBS[NEW_BLOB_ID] = {'D': [blob_centers[led_candidates_left[i][0]][0], blob_centers[led_candidates_left[i][0]][1]],
                        'U': points2D_U} 

                    for i in range(len(led_candidates_right)):
                        NEW_BLOB_ID = led_candidates_right[i][4]
                        (cx, cy, cw, ch) = blob_centers[led_candidates_right[i][0]][2]
                        cv2.rectangle(draw_frame, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 1, 1)
                        if NEW_BLOB_ID != int(Tracking_ANCHOR):
                            cv2.putText(draw_frame, f'{NEW_BLOB_ID}', (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1) 
                        points2D_D = np.array([blob_centers[led_candidates_right[i][0]][0], blob_centers[led_candidates_right[i][0]][1]], dtype=np.float64)
                        points2D_U = np.array(cv2.undistortPoints(points2D_D, camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][1])).reshape(-1, 2)
                        TEMP_BLOBS[NEW_BLOB_ID] = {'D': [blob_centers[led_candidates_right[i][0]][0], blob_centers[led_candidates_right[i][0]][1]],
                        'U': points2D_U}
                else:
                    print(f"No tracker initialized for id: {Tracking_ANCHOR}")
                    break
            

            if TRACKER_BROKEN_STATUS == DONE:
                print(f"{frame_cnt} rollback")
                continue

            # Algorithm Added
            CAMERA_INFO[f"{frame_cnt}"] = copy.deepcopy(CAMERA_INFO_STRUCTURE)
            LED_NUMBER = []
            points2D = []
            points2D_U = []
            points3D = []
            
            # TEST CODE
            points3D_PCA = []
            points3D_IQR = []            
            
            TEMP_BLOBS = OrderedDict(sorted(TEMP_BLOBS.items(), key=lambda t: t[0], reverse=True))
            for blob_id, blob_data in TEMP_BLOBS.items():
                LED_NUMBER.append(int(blob_id))
                points2D.append(blob_data['D'])
                points2D_U.append(blob_data['U'])
                
                BLOB_INFO[blob_id]['points2D_D']['greysum'].append(blob_data['D'])
                BLOB_INFO[blob_id]['points2D_U']['greysum'].append(blob_data['U'])
                BLOB_INFO[blob_id]['BLENDER']['rt']['rvec'].append(brvec_reshape)
                BLOB_INFO[blob_id]['BLENDER']['rt']['tvec'].append(btvec_reshape)
                
                points3D.append(MODEL_DATA[int(blob_id)])
                if DO_CALIBRATION_TEST == 1:
                    points3D_PCA.append(calibrated_led_data_PCA[int(blob_id)])
                    points3D_IQR.append(calibrated_led_data_IQR[int(blob_id)])
            
            print('START Pose Estimation')
            points2D = np.array(np.array(points2D).reshape(len(points2D), -1), dtype=np.float64)
            points2D_U = np.array(np.array(points2D_U).reshape(len(points2D_U), -1), dtype=np.float64)
            points3D = np.array(points3D, dtype=np.float64)
            if DO_CALIBRATION_TEST == 1:
                points3D_PCA = np.array(points3D_PCA, dtype=np.float64)
                points3D_IQR = np.array(points3D_IQR, dtype=np.float64)
            print('LED_NUMBER: ', LED_NUMBER)
            print('points2D\n', points2D)
            print('points2D_U\n', points2D_U)
            print('points3D\n', points3D)
            
            # Make CAMERA_INFO data for check rt STD
            CAMERA_INFO[f"{frame_cnt}"]['points3D'] = points3D
            if DO_CALIBRATION_TEST == 1:
                CAMERA_INFO[f"{frame_cnt}"]['points3D_PCA'] = points3D_PCA
                CAMERA_INFO[f"{frame_cnt}"]['points3D_IQR'] = points3D_IQR
                
            CAMERA_INFO[f"{frame_cnt}"]['points2D']['greysum'] = points2D
            CAMERA_INFO[f"{frame_cnt}"]['points2D_U']['greysum'] = points2D_U            
            CAMERA_INFO[f"{frame_cnt}"]['LED_NUMBER'] =LED_NUMBER

            LENGTH = len(LED_NUMBER)
            if LENGTH >= 4:
                print('PnP Solver OpenCV')
                if LENGTH >= 5:
                    METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC
                elif LENGTH == 4:
                    METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P
                INPUT_ARRAY = [
                    CAM_ID,
                    points3D,
                    points2D if undistort == 0 else points2D_U,
                    camera_matrix[CAM_ID][0] if undistort == 0 else default_cameraK,
                    camera_matrix[CAM_ID][1] if undistort == 0 else default_dist_coeffs
                ]
                ret, rvec, tvec, _ = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
                # print('PnP_Solver rvec:', rvec.flatten(), ' tvec:',  tvec.flatten())
                for blob_id in LED_NUMBER:
                    BLOB_INFO[blob_id]['OPENCV']['rt']['rvec'].append(np.array(rvec).reshape(-1, 1))
                    BLOB_INFO[blob_id]['OPENCV']['rt']['tvec'].append(np.array(tvec).reshape(-1, 1))

                image_points, _ = cv2.projectPoints(points3D,
                                                    np.array(rvec),
                                                    np.array(tvec),
                                                    camera_matrix[CAM_ID][0],
                                                    camera_matrix[CAM_ID][1])
                image_points = image_points.reshape(-1, 2)

                for point in image_points:
                    # 튜플 형태로 좌표 변환
                    pt = (int(point[0]), int(point[1]))
                    cv2.circle(draw_frame, pt, 1, (0, 0, 255), -1)

            elif LENGTH == 3:
                #P3P
                mutex.acquire()
                try:
                    print('P3P LamdaTwist')
                    points2D_U = np.array(points2D_U.reshape(len(points2D), -1))                   
                    X = np.array(points3D)   
                    x = np.hstack((points2D_U, np.ones((points2D_U.shape[0], 1))))
                    # print('normalized x\n', x)
                    poselib_result = poselib.p3p(x, X)
                    visible_detection = NOT_SET
                    for solution_idx, pose in enumerate(poselib_result):
                        colors = [(255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
                        colorstr = ['blue', 'green', 'purple', 'yellow']
                        if is_valid_pose(pose):
                            quat = pose.q
                            tvec = pose.t
                            rotm = quat_to_rotm(quat)
                            rvec, _ = cv2.Rodrigues(rotm)
                            # print("PoseLib rvec: ", rvec.flatten(), ' tvec:', tvec)                               
                            image_points, _ = cv2.projectPoints(np.array(MODEL_DATA),
                                np.array(rvec),
                                np.array(tvec),
                                camera_matrix[CAM_ID][0],
                                camera_matrix[CAM_ID][1])
                            image_points = image_points.reshape(-1, 2)
                            # print('image_points\n', image_points)
                            cam_pos, cam_dir, _ = calculate_camera_position_direction(rvec, tvec)
                            ax1.scatter(*cam_pos, c=colorstr[solution_idx], marker='o', label=f"POS{solution_idx}")
                            ax1.quiver(*cam_pos, *cam_dir, color=colorstr[solution_idx], label=f"DIR{solution_idx}", length=0.1)    
                            
                            ###############################            
                            visible_result = check_angle_and_facing(points3D, cam_pos, quat, LED_NUMBER)
                            # print('visible_result:', visible_result)
                            visible_status = SUCCESS
                            for blob_id, status in visible_result.items():
                                if status == False:
                                    visible_status = ERROR
                                    print(f"{solution_idx} pose unvisible led {blob_id}")
                                    break                                
                            if visible_status == SUCCESS:
                                visible_detection = DONE
                                for blob_id in LED_NUMBER:
                                    BLOB_INFO[blob_id]['OPENCV']['rt']['rvec'].append(np.array(rvec).reshape(-1, 1))
                                    BLOB_INFO[blob_id]['OPENCV']['rt']['tvec'].append(np.array(tvec).reshape(-1, 1))
                            ###############################
                                
                            for idx, point in enumerate(image_points):
                                # 튜플 형태로 좌표 변환
                                pt = (int(point[0]), int(point[1]))
                                if idx in LED_NUMBER:
                                    cv2.circle(draw_frame, pt, 2, (0, 0, 255), -1)
                                else:
                                    cv2.circle(draw_frame, pt, 1, colors[solution_idx], -1)
                                
                                text_offset = (5, -5)
                                text_pos = (pt[0] + text_offset[0], pt[1] + text_offset[1])
                                cv2.putText(draw_frame, str(idx), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[solution_idx], 1, cv2.LINE_AA)

                    if visible_detection == NOT_SET:
                        for blob_id in LED_NUMBER:
                            # Use an 'empty' numpy array as our NOT_SET value
                            BLOB_INFO[blob_id]['OPENCV']['rt']['rvec'].append(np.full_like(rvec, NOT_SET).reshape(-1, 1))
                            BLOB_INFO[blob_id]['OPENCV']['rt']['tvec'].append(np.full_like(tvec, NOT_SET).reshape(-1, 1))

                finally:
                    mutex.release()
            else:
                print('NOT Enough blobs')
                if AUTO_LOOP == 1:
                    frame_cnt += 1
                continue
 
        if AUTO_LOOP == 1:
            frame_cnt += 1

        cv2.imshow("Tracking", draw_frame)
        key = cv2.waitKey(8)
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
        elif key == ord('n'):
            frame_cnt += 1
            bboxes.clear()
            print('go next IMAGE')
        elif key == ord('b'):
            frame_cnt -= 1
            bboxes.clear()
            print('go back IMAGE')
    cv2.destroyAllWindows()

    data = OrderedDict()
    data['BLOB_INFO'] = BLOB_INFO
    pickle_data(WRITE, 'BLOB_INFO.pickle', data)
    data = OrderedDict()
    data['CAMERA_INFO'] = CAMERA_INFO
    pickle_data(WRITE, 'CAMERA_INFO.pickle', data)   
def remake_3d_for_blob_info(undistort):
    print('remake_3d_for_blob_info START')
    BLOB_INFO = pickle_data(READ, 'BLOB_INFO.pickle', None)['BLOB_INFO']
    REMADE_3D_INFO_B = {}
    REMADE_3D_INFO_O = {}
    # Create a pretty printer
    pp = pprint.PrettyPrinter(indent=4)
    
    print('#################### CNT for BLOB ID  ####################')
    # Iterate over the blob_ids in BLOB_INFO
    for blob_id in range(BLOB_CNT):  # Assuming BLOB_ID range from 0 to 14       
        CNT = len(BLOB_INFO[blob_id]['points2D_D']['greysum'])
        print(f"BLOB_ID: {blob_id}, CNT {CNT}")
        # pp.pprint(BLOB_INFO[blob_id])
        REMADE_3D_INFO_B[blob_id] = []
        REMADE_3D_INFO_O[blob_id] = []
        # Get the RT dictionary for the first frame
        rt_first_B = {
            'rvec': BLOB_INFO[blob_id]['BLENDER']['rt']['rvec'][0],
            'tvec': BLOB_INFO[blob_id]['BLENDER']['rt']['tvec'][0]
        }
        rt_first_O = {
            'rvec': BLOB_INFO[blob_id]['OPENCV']['rt']['rvec'][0],
            'tvec': BLOB_INFO[blob_id]['OPENCV']['rt']['tvec'][0]
        }
        points2D_D_first = [BLOB_INFO[blob_id]['points2D_D']['greysum'][0]]
        points2D_U_first = [BLOB_INFO[blob_id]['points2D_U']['greysum'][0]]
        # print('points2D_U_first: ', points2D_U_first)
        # print('rt_first_B: ', rt_first_B)

        # Iterate over the 2D coordinates for this blob_id
        # 마지막 이미지가 시작 이미지와 같아 remake에서 이상치가 발현됨
        for i in range(1, CNT):
             # Get the 2D coordinates for the first and current frame
            points2D_D_current = [BLOB_INFO[blob_id]['points2D_D']['greysum'][i]]
            points2D_U_current = [BLOB_INFO[blob_id]['points2D_U']['greysum'][i]]
            status_B = BLOB_INFO[blob_id]['BLENDER']['rt']['rvec'][i]
            status_O = BLOB_INFO[blob_id]['OPENCV']['rt']['rvec'][i]
            # Check if 'rvec' is NOT_SET for BLENDER and OPENCV separately
            if (status_B != NOT_SET).all():
                rt_current_B = {
                    'rvec': BLOB_INFO[blob_id]['BLENDER']['rt']['rvec'][i],
                    'tvec': BLOB_INFO[blob_id]['BLENDER']['rt']['tvec'][i]
                }
                if undistort == 0:
                    remake_3d_B = remake_3d_point(camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][0],
                                                rt_first_B, rt_current_B,
                                                points2D_D_first, points2D_D_current).reshape(-1, 3)
                else:
                    remake_3d_B = remake_3d_point(default_cameraK, default_cameraK,
                                                rt_first_B, rt_current_B,
                                                points2D_U_first, points2D_U_current).reshape(-1, 3)
                REMADE_3D_INFO_B[blob_id].append(remake_3d_B.reshape(-1, 3))
                
                # print('cnt: ', i)
                # print('points2D_U_current: ', points2D_U_current)
                # print('rt_current_B: ', rt_current_B)
                # print(remake_3d_B.reshape(-1, 3))
            
            if (status_O != NOT_SET).all():
                rt_current_O = {
                    'rvec': BLOB_INFO[blob_id]['OPENCV']['rt']['rvec'][i],
                    'tvec': BLOB_INFO[blob_id]['OPENCV']['rt']['tvec'][i]
                }
                if undistort == 0:
                    remake_3d_O = remake_3d_point(camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][0],
                                                rt_first_O, rt_current_O,
                                                points2D_D_first, points2D_D_current).reshape(-1, 3)
                else:
                    remake_3d_O = remake_3d_point(default_cameraK, default_cameraK,
                                                rt_first_O, rt_current_O,
                                                points2D_U_first, points2D_U_current).reshape(-1, 3)
                REMADE_3D_INFO_O[blob_id].append(remake_3d_O.reshape(-1, 3))

    file = 'REMADE_3D_INFO.pickle'
    data = OrderedDict()
    data['REMADE_3D_INFO_B'] = REMADE_3D_INFO_B
    data['REMADE_3D_INFO_O'] = REMADE_3D_INFO_O
    ret = pickle_data(WRITE, file, data)
    if ret != ERROR:
        print('data saved')
def draw_result(ax1, ax2):
    print('draw_result START')
    REMADE_3D_INFO_B = pickle_data(READ, 'REMADE_3D_INFO.pickle', None)['REMADE_3D_INFO_B']
    REMADE_3D_INFO_O = pickle_data(READ, 'REMADE_3D_INFO.pickle', None)['REMADE_3D_INFO_O']
    BA_3D = pickle_data(READ, 'BA_3D.pickle', None)['BA_3D']
    LED_INDICES = pickle_data(READ, 'BA_3D.pickle', None)['LED_INDICES']
    origin_pts = np.array(MODEL_DATA).reshape(-1, 3)

    for blob_id, data_list in REMADE_3D_INFO_B.items():
        distances_remade = []
        for data in data_list:
            point_3d = data.reshape(-1)
            distance = np.linalg.norm(origin_pts[blob_id] - point_3d)
            distances_remade.append(distance)
            if distances_remade.index(distance) == 0:
                ax1.scatter(point_3d[0], point_3d[1], point_3d[2], color='blue', alpha=0.3, marker='o', s=7,
                            label='BLENDER')
            else:
                ax1.scatter(point_3d[0], point_3d[1], point_3d[2], color='blue', alpha=0.3, marker='o', s=7)
        ax2.scatter([blob_id] * len(distances_remade), distances_remade, color='blue', alpha=0.5, marker='o', s=10,
                    label='BLENDER' if blob_id == list(REMADE_3D_INFO_B.keys())[0] else "_nolegend_")

    for blob_id, data_list in REMADE_3D_INFO_O.items():
        distances_remade = []
        for data in data_list:
            point_3d = data.reshape(-1)
            distance = np.linalg.norm(origin_pts[blob_id] - point_3d)
            distances_remade.append(distance)
            if distances_remade.index(distance) == 0:
                ax1.scatter(point_3d[0], point_3d[1], point_3d[2], color='red', alpha=0.3, marker='o', s=7,
                            label='OPENCV')
            else:
                ax1.scatter(point_3d[0], point_3d[1], point_3d[2], color='red', alpha=0.3, marker='o', s=7)
        ax2.scatter([blob_id] * len(distances_remade), distances_remade, color='red', alpha=0.5, marker='o', s=10,
                    label='OPENCV' if blob_id == list(REMADE_3D_INFO_O.keys())[0] else "_nolegend_")

    # BA 3D 정보와 origin_pts 간의 유클리드 거리를 계산하고, ax2에 그리기
    distances_ba = []
    ba_3d_dict = {}
    for i, blob_id in enumerate(LED_INDICES):
        point_3d = BA_3D[i].reshape(-1)
        distance = np.linalg.norm(origin_pts[blob_id] - point_3d)
        distances_ba.append(distance)
        if blob_id not in ba_3d_dict:
            ba_3d_dict[blob_id] = []  # 이 blob_id에 대한 리스트가 아직 없다면 새로 생성합니다.
        ba_3d_dict[blob_id].append(point_3d)
        ax1.scatter(point_3d[0], point_3d[1], point_3d[2], color='green', alpha=0.3, marker='o', s=7,
                    label='BA_3D')
    # BA에 대해서도 동일한 방식으로 scatter plot을 그립니다.
    ax2.scatter(LED_INDICES, distances_ba, color='green', alpha=0.5, marker='o', s=10, label='BA_3D')

    # 각 blob_id에 대해 PCA를 적용하고, 첫 번째 주성분에 대한 중심을 계산합니다.
    centers_ba = {}
    for blob_id, points_3d in ba_3d_dict.items():
        pca = PCA(n_components=3)  # 3차원 PCA를 계산합니다.
        pca.fit(points_3d)
        # PCA의 첫 번째 주성분의 중심을 계산합니다.
        center = pca.mean_
        centers_ba[blob_id] = center  # 이후에는 center를 원하는대로 사용하면 됩니다.

    # centers_ba에는 각 blob_id의 대표값이 저장되어 있습니다.
    print('\n')
    print('#################### PCA  ####################')
    PCA_ARRAY = []
    for blob_id, center in centers_ba.items():
        print(f"Center of PCA for blob_id {blob_id}: {center}")
        PCA_ARRAY.append(center)
    
    PCA_ARRAY_LSM = module_lsm(PCA_ARRAY)
    PCA_ARRAY_LSM = [[round(x, 8) for x in sublist] for sublist in PCA_ARRAY_LSM]
    print('PCA_ARRAY_LSM\n')
    for blob_id, points_3d in enumerate(PCA_ARRAY_LSM):
        print(f"{points_3d},")   
        
    print('\n')
    print('#################### IQR  ####################')
    IQR_ARRAY = []
    for blob_id, points_3d in ba_3d_dict.items():
        acc_blobs = points_3d.copy()
        acc_blobs_length = len(acc_blobs)
        if acc_blobs_length == 0:
            print('acc_blobs_length is 0 ERROR')
            continue

        remove_index_array = []
        med_blobs = [[], [], []]
        for blobs in acc_blobs:
            med_blobs[0].append(blobs[0])
            med_blobs[1].append(blobs[1])
            med_blobs[2].append(blobs[2])
            
        detect_outliers(med_blobs, remove_index_array)

        count = 0
        for index in remove_index_array:
            med_blobs[0].pop(index - count)
            med_blobs[1].pop(index - count)
            med_blobs[2].pop(index - count)
            count += 1

        mean_med_x = round(np.mean(med_blobs[0]), 8)
        mean_med_y = round(np.mean(med_blobs[1]), 8)
        mean_med_z = round(np.mean(med_blobs[2]), 8)
        IQR_ARRAY.append([mean_med_x, mean_med_y, mean_med_z])
        print(f"mean_med of IQR for blob_id {blob_id}: [{mean_med_x} {mean_med_y} {mean_med_z}]")           

    IQR_ARRAY_LSM = module_lsm(IQR_ARRAY)
    IQR_ARRAY_LSM = [[round(x, 8) for x in sublist] for sublist in IQR_ARRAY_LSM]
    print('IQR_ARRAY_LSM\n')
    for blob_id, points_3d in enumerate(IQR_ARRAY_LSM):
        print(f"{points_3d},")    
    
    
    # Remove duplicate labels in legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    by_label1 = dict(zip(labels1, handles1))
    ax1.legend(by_label1.values(), by_label1.keys())

    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    ax2.legend(by_label2.values(), by_label2.keys())

    file = 'RIGID_3D_TRANSFORM.pickle'
    data = OrderedDict()
    data['PCA_ARRAY_LSM'] = PCA_ARRAY_LSM
    data['IQR_ARRAY_LSM'] = IQR_ARRAY_LSM
    ret = pickle_data(WRITE, file, data)
    if ret != ERROR:
        print('data saved')

    if SHOW_PLOT == 1:
        plt.show()
def BA_3D_POINT():
    print('BA_3D_POINT START')
    BLOB_INFO = pickle_data(READ, 'BLOB_INFO.pickle', None)['BLOB_INFO']
    REMADE_3D_INFO_B = pickle_data(READ, 'REMADE_3D_INFO.pickle', None)['REMADE_3D_INFO_B']
    REMADE_3D_INFO_O = pickle_data(READ, 'REMADE_3D_INFO.pickle', None)['REMADE_3D_INFO_O']
    camera_indices = []
    point_indices = []
    estimated_RTs = []
    POINTS_2D = []
    POINTS_3D = []
    n_points = 0
    cam_id = 0

    # Iterating over each blob_id in BLOB_INFO and REMADE_3D_INFO_B
    LED_INDICES = []
    for blob_id, blob_info in BLOB_INFO.items():
        remade_3d_info = REMADE_3D_INFO_B[blob_id]
        for frame_id in range(1, len(blob_info['BLENDER']['rt']['rvec'])):
            # Adding 2D points
            if undistort == 0:
                POINTS_2D.append(blob_info['points2D_D']['greysum'][frame_id])
            else:
                POINTS_2D.append(blob_info['points2D_U']['greysum'][frame_id])
            # Adding 3D points
            POINTS_3D.append(remade_3d_info[frame_id - 1])
            # Adding RTs
            rvec = blob_info['BLENDER']['rt']['rvec'][frame_id]
            tvec = blob_info['BLENDER']['rt']['tvec'][frame_id]
            estimated_RTs.append((rvec.ravel(), tvec.ravel()))

            # Adding camera id
            camera_indices.append(cam_id)
            # Adding point index
            point_indices.append(cam_id)
            LED_INDICES.append(blob_id)
            cam_id += 1
        n_points += (len(blob_info['BLENDER']['rt']['rvec']) - 1)

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
                                                 camera_matrix[CAM_ID][0] if undistort == 0 else default_cameraK,
                                                 camera_matrix[CAM_ID][1] if undistort == 0 else default_dist_coeffs)
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
    
    print('\n')
    print('#################### BA  ####################')
    print('n_points', n_points)
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-6, method='trf',
                        args=(n_points, camera_indices, point_indices, POINTS_2D, camera_params, camera_matrix))

    # You are only optimizing points, so the result only contains point data
    n_points_3d = res.x.reshape((n_points, 3))
    # print("Optimized 3D points: ", n_points_3d, ' ', len(n_points_3d))
    file = 'BA_3D.pickle'
    data = OrderedDict()
    data['BA_3D'] = n_points_3d
    data['LED_INDICES'] = LED_INDICES
    ret = pickle_data(WRITE, file, data)
    if ret != ERROR:
        print('data saved')
def Check_Calibration_data_combination():
    print('Check_Calibration_data_combination START')
    CAMERA_INFO = pickle_data(READ, 'CAMERA_INFO.pickle', None)['CAMERA_INFO']       
    def reprojection_error(points3D, points2D, rvec, tvec, camera_k, dist_coeff):        
        points2D_reprojection, _ = cv2.projectPoints(points3D, np.array(rvec), np.array(tvec), camera_k, dist_coeff)
        # Squeeze the points2D_reprojection to match the dimensionality of points2D
        points2D_reprojection = points2D_reprojection.squeeze()
        RER = np.average(np.linalg.norm(points2D - points2D_reprojection, axis=1))
        # print('points2D:\n', points2D)
        # print('points2D_reprojection:\n', points2D_reprojection)
        # print('RER:', RER)
        return RER
    def STD_Analysis(points3D_data, label):
        print(f"{points3D_data}")
        frame_counts = []
        rvec_std_arr = []
        tvec_std_arr = []
        reproj_err_rates = []
        
        for frame_cnt, cam_data in CAMERA_INFO.items():           
            
            rvec_list = []
            tvec_list = []
            reproj_err_list = []
            
            LED_NUMBER = cam_data['LED_NUMBER']
            points3D = cam_data[points3D_data]
            points2D = cam_data['points2D']['greysum']
            points2D_U = cam_data['points2D_U']['greysum']
            
            LENGTH = len(LED_NUMBER)
            
            if LENGTH >= 4:
                for r in range(4, LENGTH + 1 if FULL_COMBINATION_SEARCH == 1 else 5):
                    for comb in combinations(range(LENGTH), r):
                        for perm in permutations(comb):
                            points3D_perm = points3D[list(perm), :]
                            points2D_perm = points2D[list(perm), :]
                            points2D_U_perm = points2D_U[list(perm), :]
                            if r >= 5:
                                METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC
                            elif r == 4:
                                METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P
                            INPUT_ARRAY = [
                                CAM_ID,
                                points3D_perm,
                                points2D_perm if undistort == 0 else points2D_U_perm,
                                camera_matrix[CAM_ID][0] if undistort == 0 else default_cameraK,
                                camera_matrix[CAM_ID][1] if undistort == 0 else default_dist_coeffs
                            ]
                            _, rvec, tvec, _ = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
                            
                            rvec_list.append(np.linalg.norm(rvec))
                            tvec_list.append(np.linalg.norm(tvec))

                            RER = reprojection_error(points3D_perm,
                                                    points2D_perm,
                                                    rvec, tvec,
                                                    camera_matrix[CAM_ID][0],
                                                    camera_matrix[CAM_ID][1])                        
                            reproj_err_list.append(RER)

                        frame_counts.append(frame_cnt)
                        rvec_std = np.std(rvec_list)
                        tvec_std = np.std(tvec_list)
                        reproj_err_rate = np.mean(reproj_err_list)
                        
                        rvec_std_arr.append(rvec_std)
                        tvec_std_arr.append(tvec_std)
                        reproj_err_rates.append(reproj_err_rate)

        return frame_counts, rvec_std_arr, tvec_std_arr, reproj_err_rates, label


    
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))

    points3D_datas = ['points3D', 'points3D_PCA', 'points3D_IQR']
    colors = ['b', 'g', 'r']
    summary_text = ""

    for idx, points3D_data in enumerate(points3D_datas):
        frame_counts, rvec_std_arr, tvec_std_arr, reproj_err_rates, label = STD_Analysis(points3D_data, points3D_data)

        axs[0].plot(frame_counts, rvec_std_arr, colors[idx]+'-', label=f'rvec std {label}')
        axs[0].plot(frame_counts, tvec_std_arr, colors[idx]+'--', label=f'tvec std {label}')
        axs[1].plot(frame_counts, reproj_err_rates, colors[idx], label=f'Reprojection error rate {label}')
        
        # Calculate and store the average and standard deviation for each data set
        avg_rvec_std = np.mean(rvec_std_arr)
        std_rvec_std = np.std(rvec_std_arr)
        avg_tvec_std = np.mean(tvec_std_arr)
        std_tvec_std = np.std(tvec_std_arr)
        avg_reproj_err = np.mean(reproj_err_rates)
        std_reproj_err = np.std(reproj_err_rates)

        summary_text += f"== {label} ==\n"
        summary_text += f"Rvec Std: Mean = {avg_rvec_std:.2f}, Std = {std_rvec_std:.2f}\n"
        summary_text += f"Tvec Std: Mean = {avg_tvec_std:.2f}, Std = {std_tvec_std:.2f}\n"
        summary_text += f"Reproj Err: Mean = {avg_reproj_err:.2f}, Std = {std_reproj_err:.2f}\n"
        summary_text += "\n"

    axs[0].legend()
    axs[0].set_xlabel('frame_cnt')
    axs[0].set_ylabel('std')
    axs[0].set_title('Standard Deviation of rvec and tvec Magnitude per Frame')

    axs[1].legend()
    axs[1].set_xlabel('frame_cnt')
    axs[1].set_ylabel('error rate')
    axs[1].set_title('Mean Reprojection Error Rate per Frame')
    
    axs[2].axis('off')  # Hide axes for the text plot
    axs[2].text(0, 0, summary_text, fontsize=10)

    # Reducing the number of X-ticks to avoid crowding
    for ax in axs[:2]:
        ax.set_xticks(ax.get_xticks()[::5])

    plt.subplots_adjust(hspace=0.3)  # Add space between subplots
    plt.show()
def Check_Calibration_data():
    CAMERA_INFO = pickle_data(READ, 'CAMERA_INFO.pickle', None)['CAMERA_INFO']       
    def reprojection_error(points3D, points2D, rvec, tvec, camera_k, dist_coeff):        
        points2D_reprojection, _ = cv2.projectPoints(points3D, np.array(rvec), np.array(tvec), camera_k, dist_coeff)
        # Squeeze the points2D_reprojection to match the dimensionality of points2D
        points2D_reprojection = points2D_reprojection.squeeze()
        RER = np.average(np.linalg.norm(points2D - points2D_reprojection, axis=1))
        # print('points2D:\n', points2D)
        # print('points2D_reprojection:\n', points2D_reprojection)
        # print('RER:', RER)
        return RER

    def STD_Analysis(points3D_data, label):
        frame_counts_rvec = []
        frame_counts_tvec = []
        frame_counts_reproj_err = []
        rvec_std_arr = []
        tvec_std_arr = []
        reproj_err_rates = []

        for frame_cnt, cam_data in CAMERA_INFO.items():
            LED_NUMBER = cam_data['LED_NUMBER']
            points3D = cam_data[points3D_data]
            points2D = cam_data['points2D']['greysum']
            points2D_U = cam_data['points2D_U']['greysum']

            LENGTH = len(LED_NUMBER)

            if LENGTH >= 4:
                print('PnP Solver OpenCV')
                if LENGTH >= 5:
                    METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC
                elif LENGTH == 4:
                    METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P

                INPUT_ARRAY = [
                    CAM_ID,
                    points3D,
                    points2D if undistort == 0 else points2D_U,
                    camera_matrix[CAM_ID][0] if undistort == 0 else default_cameraK,
                    camera_matrix[CAM_ID][1] if undistort == 0 else default_dist_coeffs
                ]
                _, rvec, tvec, _ = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)

                RER = reprojection_error(points3D,
                                         points2D,
                                         rvec, tvec,
                                         camera_matrix[CAM_ID][0],
                                         camera_matrix[CAM_ID][1])
 

                frame_counts_rvec.append(frame_cnt)
                rvec_std_arr.append(np.linalg.norm(rvec))
                                    
                frame_counts_tvec.append(frame_cnt)
                tvec_std_arr.append(np.linalg.norm(tvec))

                reproj_err_rates.append(RER)
                frame_counts_reproj_err.append(frame_cnt)
                

        return frame_counts_rvec, rvec_std_arr, frame_counts_tvec, tvec_std_arr, frame_counts_reproj_err, reproj_err_rates, label

    fig, axs = plt.subplots(3, 1, figsize=(15, 15))

    points3D_datas = ['points3D','points3D_PCA','points3D_IQR']
    colors = ['b', 'g', 'r']
    summary_text = ""

    for idx, points3D_data in enumerate(points3D_datas):
        frame_counts_rvec, rvec_std_arr, frame_counts_tvec, tvec_std_arr, frame_counts_reproj_err, reproj_err_rates, label = STD_Analysis(points3D_data, points3D_data)

        axs[0].plot(frame_counts_rvec, rvec_std_arr, colors[idx]+'-', label=f'rvec std {label}')
        axs[0].plot(frame_counts_tvec, tvec_std_arr, colors[idx]+'--', label=f'tvec std {label}')
        axs[1].plot(frame_counts_reproj_err, reproj_err_rates, colors[idx], label=f'Reprojection error rate {label}')

        # Calculate and store the average and standard deviation for each data set
        avg_rvec_std = np.mean(rvec_std_arr)
        std_rvec_std = np.std(rvec_std_arr)
        avg_tvec_std = np.mean(tvec_std_arr)
        std_tvec_std = np.std(tvec_std_arr)
        avg_reproj_err = np.mean(reproj_err_rates)
        std_reproj_err = np.std(reproj_err_rates)

        summary_text += f"== {label} ==\n"
        summary_text += f"Rvec Std: Mean = {avg_rvec_std:.2f}, Std = {std_rvec_std:.2f}\n"
        summary_text += f"Tvec Std: Mean = {avg_tvec_std:.2f}, Std = {std_tvec_std:.2f}\n"
        summary_text += f"Reproj Err: Mean = {avg_reproj_err:.2f}, Std = {std_reproj_err:.2f}\n"
        summary_text += "\n"

    axs[0].legend()
    axs[0].set_xlabel('frame_cnt')
    axs[0].set_ylabel('std')
    axs[0].set_title('Standard Deviation of rvec and tvec Magnitude per Frame')

    axs[1].legend()
    axs[1].set_xlabel('frame_cnt')
    axs[1].set_ylabel('error rate')
    axs[1].set_title('Mean Reprojection Error Rate per Frame')

    axs[2].axis('off')  # Hide axes for the text plot
    axs[2].text(0, 0, summary_text, fontsize=10)

    # Reducing the number of X-ticks to avoid crowding
    for ax in axs[:2]:
        ax.set_xticks(ax.get_xticks()[::5])

    plt.subplots_adjust(hspace=0.3)  # Add space between subplots
    plt.show()
def init_plot():
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

    led_number = len(MODEL_DATA)
    ax2.set_title('distance')
    ax2.set_xlim([0, led_number])  # x축을 LED 번호의 수 만큼 설정합니다. 이 경우 14개로 설정됩니다.
    ax2.set_xticks(range(led_number))  # x축에 표시되는 눈금을 LED 번호의 수 만큼 설정합니다.
    ax2.set_xticklabels(range(led_number))  # x축에 표시되는 눈금 라벨을 LED 번호의 수 만큼 설정합니다.

    origin_pts = np.array(MODEL_DATA).reshape(-1, 3)
    ax1.set_title('3D plot')    
    # ax1.scatter(origin_pts[:, 0], origin_pts[:, 1], origin_pts[:, 2], color='gray', alpha=1.0, marker='o', s=10, label='ORIGIN')
    
    ax1.scatter(0, 0, 0, marker='o', color='k', s=20)
    ax1.set_xlim([-0.2, 0.2])
    ax1.set_xlabel('X')
    ax1.set_ylim([-0.2, 0.2])
    ax1.set_ylabel('Y')
    ax1.set_zlim([-0.2, 0.2])
    ax1.set_zlabel('Z')
    scale = 1.5
    f = zoom_factory(ax1, base_scale=scale)
    
    return ax1, ax2
def init_coord_json(file):
    print(init_coord_json.__name__)
    try:
        json_file = open(f'{file}')
        jsonObject = json.load(json_file)
        model_points = jsonObject.get('TrackedObject').get('ModelPoints')
        pts = [0 for i in range(len(model_points))]
        dir = [0 for i in range(len(model_points))]
        for data in model_points:
            idx = data.split('Point')[1]
            x = model_points.get(data)[0]
            y = model_points.get(data)[1]
            z = model_points.get(data)[2]
            u = model_points.get(data)[3]
            v = model_points.get(data)[4]
            w = model_points.get(data)[5]
            r1 = model_points.get(data)[6]
            r2 = model_points.get(data)[7]
            r3 = model_points.get(data)[8]

            pts[int(idx)] = np.array([x, y, z])
            dir[int(idx)] = np.array([u, v, w])

            # print(''.join(['{ .pos = {{', f'{x}', ',', f'{y}', ',', f'{z}',
            #                     ' }}, .dir={{', f'{u}', ',', f'{v}', ',', f'{w}', ' }}, .pattern=', f'{idx}', '},']))
    except:
        print('exception')
        traceback.print_exc()
    finally:
        print('done')
    return pts, dir
def show_calibrate_data(model_data, direction):    
    # 3D plot 생성
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 점들을 plot에 추가
    ax.scatter(model_data[:, 0], model_data[:, 1], model_data[:, 2])

    # 각 점에 대한 인덱스를 추가
    for i in range(model_data.shape[0]):
        ax.text(model_data[i, 0], model_data[i, 1], model_data[i, 2], str(i))

    # 각 점에서 방향 벡터를 나타내는 화살표를 그림
    for i in range(model_data.shape[0]):
        ax.quiver(model_data[i, 0], model_data[i, 1], model_data[i, 2], 
                direction[i, 0], direction[i, 1], direction[i, 2],
                color='b',length=0.01)


    ax.scatter(0, 0, 0, marker='o', color='k', s=20)
    ax.set_xlim([-0.2, 0.2])
    ax.set_xlabel('X')
    ax.set_ylim([-0.2, 0.2])
    ax.set_ylabel('Y')
    ax.set_zlim([-0.2, 0.2])
    ax.set_zlabel('Z')
    scale = 1.5
    f = zoom_factory(ax, base_scale=scale)

    plt.show()


def recover_pose_essential_test():
    print('recover_pose_essential_test START')
    CAMERA_INFO = pickle_data(READ, 'CAMERA_INFO.pickle', None)['CAMERA_INFO']

    # for frame_cnt, cam_data in CAMERA_INFO.items():
    #     LED_NUMBER = cam_data['LED_NUMBER']
    #     print(LED_NUMBER)

    def recover_3d_points_combinations(groups, K):
        recovered_3d_points = {}
        for key, group in groups.items():
            frames = list(group["frames"].keys())
            if len(key) < 5:
                continue
            print('key: ', key, ' len: ', len(frames))
            frame1 = min(frames)
            frame2 = max(frames)
            points1_2d = np.array(group["frames"][frame1]["points2D_U"])
            points2_2d = np.array(group["frames"][frame2]["points2D_U"])
            print('points1_2d:' , points1_2d)
            print('points2_2d:' , points2_2d)

            E, mask = cv2.findEssentialMat(points1_2d, points2_2d, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

            # Check if Essential Matrix is None
            if E is None:
                continue

            # Check the shape and validity of Essential Matrix
            if E.shape != (3, 3) or np.isnan(E).any() or np.isinf(E).any():
                continue
                    
            _, R, t, _ = cv2.recoverPose(E, points1_2d, points2_2d, K)
            
            R1 = np.eye(3)
            t1 = np.zeros((3, 1))
            R2 = R
            t2 = t
            points_3d_homogeneous = cv2.triangulatePoints(np.hstack((R1, t1)), np.hstack((R2, t2)), points1_2d.T, points2_2d.T)
            points_3d = points_3d_homogeneous[:3, :] / points_3d_homogeneous[3, :]
            combination_key = (key, (frame1, frame2))
            recovered_3d_points[combination_key] = points_3d.T
            print('points_3d.T:',points_3d.T)
        return recovered_3d_points

    def group_points_by_led_sequence(CAMERA_INFO):
        groups = {}
        for frame_cnt, cam_data in CAMERA_INFO.items():
            led_numbers = cam_data['LED_NUMBER']
            points2D = cam_data['points2D']['greysum']
            points2D_U = cam_data['points2D_U']['greysum']

            if int(frame_cnt) > 8:
                break
            # LED 번호의 순서를 키로 사용
            key = tuple(led_numbers)

            if key not in groups:
                groups[key] = {"frames": {}}

            if frame_cnt not in groups[key]["frames"]:
                groups[key]["frames"][frame_cnt] = {"points2D": [], "points2D_U": []}

            groups[key]["frames"][frame_cnt]["points2D"].extend(points2D)
            groups[key]["frames"][frame_cnt]["points2D_U"].extend(points2D_U)
        return groups
    # 위의 함수를 사용해서 그룹 생성
    groups = group_points_by_led_sequence(CAMERA_INFO)
    recover_points = recover_3d_points_combinations(groups, camera_matrix[0][0])
    # # 출력해보기
    # for key, group in groups.items():
    #     print(f"LED Sequence: {key}")
    #     for frame, data in group["frames"].items():
    #         print(f"Frame: {frame}")
    #         print(f"2D Points: {data['points2D']}")
    #         print(f"Undistorted 2D Points: {data['points2D_U']}")
    #         print("\n")
    recover_points_list = [point for key, point in recover_points.items()]
    recover_points_array = np.array(recover_points_list)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    ax1.scatter(recover_points_array[:, 0], recover_points_array[:, 1], recover_points_array[:, 2], color='gray', alpha=1.0, marker='o', s=10, label='ORIGIN')
    
    ax1.scatter(0, 0, 0, marker='o', color='k', s=20)
    ax1.set_xlim([-0.2, 0.2])
    ax1.set_xlabel('X')
    ax1.set_ylim([-0.2, 0.2])
    ax1.set_ylabel('Y')
    ax1.set_zlim([-0.2, 0.2])
    ax1.set_zlabel('Z')
    scale = 1.5
    f = zoom_factory(ax1, base_scale=scale)
    plt.show()

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(os.getcwd())
    MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/{controller_name}.json"))    
    BLOB_CNT = len(MODEL_DATA)
    print('PTS')
    for i, leds in enumerate(MODEL_DATA):
        print(f"{np.array2string(leds, separator=', ')},")
    print('DIR')
    for i, dir in enumerate(DIRECTION):
        print(f"{np.array2string(dir, separator=', ')},")
    # show_calibrate_data(np.array(MODEL_DATA), np.array(DIRECTION))

    # ax1, ax2 = init_plot()
    # bboxes = blob_setting(script_dir)
    # gathering_data_single(ax1, script_dir, bboxes)
    # remake_3d_for_blob_info(undistort)
    # BA_3D_POINT()
    # draw_result(ax1, ax2)
    # Check_Calibration_data_combination()
    recover_pose_essential_test()
    
    print('\n\n')
    print('########## DONE ##########')