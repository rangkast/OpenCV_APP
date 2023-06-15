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
    image_files = sorted(glob.glob('./tmp/render/*.png'))
    frame_0 = cv2.imread(image_files[0])
    if frame_0 is None:
        print("Cannot read the first image")
        cv2.destroyAllWindows()
        exit()

    _, frame_0 = cv2.threshold(cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                               cv2.THRESH_TOZERO)
    draw_frame = frame_0.copy()
    prev_data = rw_json_data(READ, path)
    if prev_data != ERROR:
        trackers = prev_data
        init_trackers(trackers, frame_0)
    else:
        trackers = {}
    blob_area_0 = detect_led_lights(draw_frame, 5, 5, 500)
    cv2.namedWindow('image')
    partial_click_event = functools.partial(click_event, frame_0=frame_0, blob_area_0=blob_area_0, trackers=trackers)
    cv2.setMouseCallback('image', partial_click_event)
    while True:
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('c'):
            trackers = {}
            draw_frame = frame_0.copy()
        elif key & 0xFF == ord('s'):
            save_data = {id: {'bbox': data['bbox']} for id, data in trackers.items()}
            rw_json_data(WRITE, path, save_data)
            init_trackers(trackers, frame_0)
        elif key & 0xFF == 27:
            print('ESC pressed')
            cv2.destroyAllWindows()
            return ERROR

        draw_blobs_and_ids(draw_frame, blob_area_0, trackers)
        cv2.imshow('image', draw_frame)
    cv2.destroyAllWindows()

    THRESHOLD_DISTANCE = 10  # Define your threshold distance here
    previous_positions = {}  # Store the previous positions of the trackers
    frame_cnt = 0

    last_led_number = int(next(iter(trackers)))
    print('last_led_number:', last_led_number)

    while frame_cnt < len(image_files):
        frame_0 = cv2.imread(image_files[frame_cnt])
        if frame_0 is None or frame_0.size == 0:
            print(f"Failed to load {image_files[frame_cnt]}, frame_cnt:{frame_cnt}")
            continue

        draw_frame = frame_0.copy()
        _, frame_0 = cv2.threshold(cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                   cv2.THRESH_TOZERO)
        filename = os.path.basename(image_files[frame_cnt])
        cv2.putText(draw_frame, filename, (draw_frame.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    1)
        LED_NUMBER = []
        CAMERA_INFO[f"{frame_cnt}"] = copy.deepcopy(CAMERA_INFO_STRUCTURE)
        blob_area_0 = detect_led_lights(frame_0, 5, 5, 500)
        blob_centers = []
        for blob_id, bbox in enumerate(blob_area_0):
            (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
            gcx, gcy = find_center(frame_0, (x, y, w, h))
            blob_centers.append((gcx, gcy))
            cv2.putText(draw_frame, f'{blob_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 1)

        tracking_success = False
        # Make a copy of the trackers dictionary
        trackers_copy = trackers.copy()

        print('frame_cnt:', frame_cnt)
        print('trackers_copy:', trackers_copy)
        print('1.previous_positions:', previous_positions)
        for id, data in trackers_copy.items():
            ok, bbox = data['tracker'].update(frame_0)
            IDX = int(id)
            if ok:
                (tx, ty, tw, th) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                cv2.rectangle(draw_frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 1, 1)
                cv2.putText(draw_frame, f'{IDX}', (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (0, 255, 0), 1)
                tcx, tcy = find_center(frame_0, (tx, ty, tw, th))
                # Check if the current position is far from the previous position
                if id in previous_positions:
                    dx = previous_positions[id][0] - tcx
                    dy = previous_positions[id][1] - tcy
                    euclidean_distance = math.sqrt(dx ** 2 + dy ** 2)
                    if euclidean_distance > THRESHOLD_DISTANCE:
                        del trackers[id]
                        del previous_positions[id]

                        PREV_LED_NUMBER = CAMERA_INFO[f"{frame_cnt - 1}"]['led_num']
                        (last_led_number_area, last_led_number, last_points) = copy.deepcopy(PREV_LED_NUMBER[2])

                        print('number,area:', last_led_number, last_led_number_area, euclidean_distance, last_points)
                        if last_led_number >= 0:

                            print('blob_centers:', blob_centers)
                            # Get the coordinates of the last point
                            last_x, last_y = last_points
                            # Calculate Euclidean distances from the last point to all blob centers
                            distances = [np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2) for x, y in blob_centers]
                            # Get the index of the blob center with the smallest distance
                            closest_blob_index = np.argmin(distances)
                            # Get the bounding box of the closest blob
                            led_bbox = blob_area_0[closest_blob_index]

                            trackers[last_led_number] = {'bbox': led_bbox, 'tracker': None}
                            init_trackers(trackers, frame_0)

                        print('trackers:', trackers)
                        ntcx, ntcy = find_center(frame_0, led_bbox)
                        previous_positions[last_led_number] = (ntcx, ntcy)
                        # print('2.previous_positions:', previous_positions)

                        break

                # Save the current position for the next frame
                previous_positions[id] = (tcx, tcy)
                tracking_success = True
                print(f"IDX {IDX}, last_led_number {last_led_number}")
                if IDX == last_led_number:  # Using the dynamic last LED number
                    # Find blobs at the same position as the tracker
                    same_position_blob = [
                        (idx, math.sqrt((blob_centers[idx][0] - tcx) ** 2 + (blob_centers[idx][1] - tcy) ** 2)) for
                        idx in range(len(blob_area_0))]
                    same_position_blob.sort(key=lambda x: x[1])

                    print('same_position_blob:', same_position_blob)
                    if same_position_blob:
                        _, dist_tracker_blob = same_position_blob[0]
                        # if the distance is within a certain threshold
                        if dist_tracker_blob < 1:  # assuming 1 as threshold distance
                            # Find blobs to the right of the tracker and sort them by their distances from the tracker
                            led_candidates = [(idx, blob_centers[idx][0],
                                               math.sqrt((blob_centers[idx][0] - tcx) ** 2 + (
                                                           blob_centers[idx][1] - tcy) ** 2))
                                              for idx in range(len(blob_area_0)) if blob_centers[idx][0] > tcx]
                            led_candidates.sort(key=lambda x: x[2])

                            # Assign LED numbers in descending order from the closest blob
                            for i in range(min(3, len(led_candidates))):
                                if last_led_number - i - 1 >= 0:  # Ensuring the LED number doesn't go below 11
                                    LED_NUMBER.append((blob_area_0[led_candidates[i][0]],
                                                       last_led_number - i - 1, blob_centers[i]))  # LEDs to the right of the tracker
                    else:
                        print('same_pos_not_found:', same_position_blob)

        for led_area, number, points in LED_NUMBER:
            (x, y, w, h) = led_area
            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 0, 255), 1, 1)  # Draw box in red for selected LEDs
            cv2.putText(draw_frame, f'{number}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)  # Draw LED number in red

        if not tracking_success:  # If no trackers are successfully updated, break the loop
            continue

        print('LED_NUMBER:', LED_NUMBER)
        CAMERA_INFO[f"{frame_cnt}"]['led_num'] = LED_NUMBER

        # print('camera_info\n', CAMERA_INFO[f"{frame_cnt}"])
        cv2.imshow("Tracking", draw_frame)
        frame_cnt += 1
        key = cv2.waitKey(100)
        # Exit if ESC key is
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == 27:
            print('ESC pressed')
            cv2.destroyAllWindows()
            return ERROR
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print(os.getcwd())
    print('origin')
    for i, leds in enumerate(origin_led_data):
        print(f"{i}, {leds}")
    print('target')
    for i, leds in enumerate(target_led_data):
        print(f"{i}, {leds}")

    gathering_data()
