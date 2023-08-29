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
from itertools import combinations, permutations
from matplotlib.ticker import MaxNLocator
from collections import defaultdict
from data_class import *
import torch
import torchvision
import kornia as K
import time
from sklearn.cluster import AgglomerativeClustering
from numba import jit, float64, int32
# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
# Add the directory containing poselib to the module search path
print(script_dir)

sys.path.append(os.path.join(script_dir, '../../../../EXTERNALS'))
# poselib only working in LINUX or WSL (window)
import poselib
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(f"{script_dir}../../../../connection"))))


CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
READ = 0
WRITE = 1
SUCCESS = 0
ERROR = -1
DONE = 1
NOT_SET = -1

TOP = 0
BOTTOM = 1

PLUS = 0
MINUS = 1


camera_matrix = [
    [np.array([[715.159, 0.0, 650.741],
               [0.0, 715.159, 489.184],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.075663], [-0.027738], [0.007440], [-0.000961]], dtype=np.float64)],
]

default_dist_coeffs = np.zeros((4, 1))
default_cameraK = np.eye(3).astype(np.float64)


CAMERA_INFO = {}
CAMERA_INFO_STRUCTURE = {
    'LED_NUMBER': [],
    'ANGLE': NOT_SET,
    'points2D': {'greysum': [], 'opencv': [], 'blender': []},
    'points2D_U': {'greysum': [], 'opencv': [], 'blender': []},
    'points3D': [],
    'points3D_origin': [],
    'points3D_legacy': [],
    'points3D_PCA': [],
    'points3D_IQR': [],
    'BLENDER': {'rt': {'rvec': [], 'tvec': []}, 'status': NOT_SET},
    'OPENCV': {'rt': {'rvec': [], 'tvec': []}, 'status': NOT_SET},
    'BA_RT': {'rt': {'rvec': [], 'tvec': []}, 'status': NOT_SET},
    'bboxes':[]
}

BLOB_INFO = {}
BLOB_INFO_STRUCTURE = {
    'points2D_D': {'greysum': []},
    'points2D_U': {'greysum': []},
    'BLENDER': {'rt': {'rvec': [], 'tvec': []}, 'status': []},
    'OPENCV': {'rt': {'rvec': [], 'tvec': []}, 'status': []},
    'BA_RT': {'rt': {'rvec': [], 'tvec': []}, 'status': []},
}
UP = 0
DOWN = 1
MOVE = 2
RECTANGLE = 0
CIRCLE = 1

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
            if y < 0 or y >= CAP_PROP_FRAME_HEIGHT or x < 0 or x >= CAP_PROP_FRAME_WIDTH:
                continue
            x_sum += x * frame[y][x]
            t_sum += frame[y][x]
            if frame[y][x] > 0:
                m_count += 1

    for x in range(X, X + W):
        for y in range(Y, Y + H):
            if y < 0 or y >= CAP_PROP_FRAME_HEIGHT or x < 0 or x >= CAP_PROP_FRAME_WIDTH:
                continue
            y_sum += y * frame[y][x]

    if t_sum != 0:
        g_c_x = x_sum / t_sum
        g_c_y = y_sum / t_sum

    if g_c_x == 0 or g_c_y == 0:
        return 0, 0, 0

    return g_c_x, g_c_y, m_count
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

        # Use contourArea to get the actual area of the contour
        area = cv2.contourArea(contour)

        # Check if the area of the contour is within the specified range
        blob_info.append((x, y, w, h))

    return blob_info
def point_in_bbox(x, y, bbox):
    return bbox[0] <= x <= bbox[0] + bbox[2] and bbox[1] <= y <= bbox[1] + bbox[3]
def draw_blobs_and_ids(frame, blobs, bboxes):
    for bbox in blobs:
        (x, y, w, h) = bbox[2]
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 1, 1)
        
    for box in bboxes:
        cv2.putText(frame, f"{box['idx']}", (int(box['bbox'][0]), int(box['bbox'][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 1)
        cv2.rectangle(frame, (int(box['bbox'][0]), int(box['bbox'][1])), (int(box['bbox'][0]) + int(box['bbox'][2]),
                                                                           int(box['bbox'][1]) + int(box['bbox'][3])), (0, 255, 0), 1, 1)
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
                # json.dump(data, wdata, separators=(',', ':'))
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
    

def click_event(event, x, y, flags, param, frame, blob_area_0, bboxes, POS=NOT_SET):
    if event == cv2.EVENT_LBUTTONDOWN:
        down_in_box = NOT_SET
        # print(f"EVENT_LBUTTONDOWN {x} {y}")
        for i, bbox in enumerate(blob_area_0):
            if point_in_bbox(x, y, bbox[2]):
                input_number = input('Please enter ID for this bbox: ')
                bboxes.append({'idx': input_number, 'bbox': bbox[2]})
                draw_blobs_and_ids(frame, blob_area_0, bboxes)
                down_in_box = DONE
        if down_in_box == NOT_SET and POS != NOT_SET:
            if POS['status'] == UP or POS['status'] == NOT_SET:
                POS['start'] = [x, y]
                POS['status'] = DOWN
    elif event == cv2.EVENT_MOUSEMOVE and POS != NOT_SET:
        if POS['status'] == DOWN or POS['status'] == MOVE:
            POS['move'] = [x, y]
            POS['status'] = MOVE            

    elif event == cv2.EVENT_LBUTTONUP and POS != NOT_SET:
        # print(f"EVENT_LBUTTONUP {x} {y}")
        POS['status'] = UP
            

def read_camera_log(file_path):
    if not os.path.exists(file_path):
        return "ERROR"

    with open(file_path, 'r') as file:
        lines = file.readlines()

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

def quat_to_rotm(q):
    qw, qx, qy, qz = q
    qx2, qy2, qz2 = qx * qx, qy * qy, qz * qz
    qxqy, qxqz, qyqz = qx * qy, qx * qz, qy * qz
    qxqw, qyqw, qzqw = qx * qw, qy * qw, qz * qw

    return np.array([[1 - 2 * (qy2 + qz2), 2 * (qxqy - qzqw), 2 * (qxqz + qyqw)],
                     [2 * (qxqy + qzqw), 1 - 2 * (qx2 + qz2), 2 * (qyqz - qxqw)],
                     [2 * (qxqz - qyqw), 2 * (qyqz + qxqw), 1 - 2 * (qx2 + qy2)]])

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
def show_calibrate_data(model_data, direction, **kwargs):
    Candidates_points = kwargs.get('TARGET')
    # 3D plot 생성
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if Candidates_points is not None:
        ax.scatter(Candidates_points[:, 0], Candidates_points[:, 1], Candidates_points[:, 2], color='red')
        for i in range(Candidates_points.shape[0]):
            ax.text(Candidates_points[i, 0], Candidates_points[i, 1], Candidates_points[i, 2], str(i), color='red')

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
            if ellipse[1][0] > 0 and ellipse[1][1] > 0:  # width and height must be positive
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
        found, img_with_contours, draw_frame_with_shapes = find_circles_or_ellipses(img.copy(), draw_frame_cropped.copy())
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
def rigid_transform_2D(A, B):
    assert len(A) == len(B)
    num_pts = len(A)
    cx_A = np.mean(A[:,0])
    cy_A = np.mean(A[:,1])
    cx_B = np.mean(B[:,0])
    cy_B = np.mean(B[:,1])
    
    # 이동 거리 계산
    d = [[cx_B-cx_A], [cy_B-cy_A]]

    # 회전을 계산하기 위해 이동 거리만큼 먼저 평행 이동
    Ashift = A - np.tile([cx_A, cy_A], (num_pts, 1))
    Bshift = B - np.tile([cx_B, cy_B], (num_pts, 1))

    # H 행렬 계산
    H = Ashift.T @ Bshift
    
    # 특이값 분해를 이용하여 R 행렬(회전) 계산
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    return R, d

def module_lsm_2D(MODEL_DATA, blob_data):
    print(module_lsm_2D.__name__)
    print('MODEL_DATA\n', MODEL_DATA)
    print('blob_data\n', blob_data)

    # PCA 객체를 생성하고 3D 점을 2D로 투영
    pca = PCA(n_components=2)
    MODEL_DATA_2D = pca.fit_transform(MODEL_DATA)
    blob_data_2D = pca.transform(blob_data)  # 같은 PCA 변환을 사용

    origin_pts = []
    before_pts = []
    after_pts = []
    led_array = []
    for led_id, points in enumerate(blob_data_2D):
        led_array.append(led_id)
        origin_pts.append(MODEL_DATA_2D[led_id])
        before_pts.append(points)

    # 2D 변환 계산
    R, t = rigid_transform_2D(np.array(before_pts), np.array(origin_pts))

    # 변환 적용
    for point in before_pts:
        new_point = R @ point + t
        after_pts.append(new_point)

    # 2D 점을 원래의 3D 공간으로 변환
    after_pts_3D = pca.inverse_transform(after_pts)

    print('after_pts_3D\n', after_pts_3D)
    # 첫 번째 열은 변환 전 좌표, 두 번째 열은 변환 후 좌표
    before_transformation = [point[0] for point in after_pts_3D]
    after_transformation = [point[1] for point in after_pts_3D]

    # Numpy 배열로 변환 (필요에 따라 사용)
    before_transformation = np.array(before_transformation)
    after_transformation = np.array(after_transformation)

    return after_transformation

# rigid_transform_2D 함수는 위에서 설명한 대로 정의합니다.

def module_lsm_3D(MODEL_DATA, blob_data):
    print(module_lsm_3D.__name__)

    # print('MODEL_DATA\n', MODEL_DATA)
    # print('blob_data\n', blob_data)

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
def rotation_matrix_to_quaternion(r):
    q0 = np.sqrt(1 + r[0, 0] + r[1, 1] + r[2, 2]) / 2
    q1 = (r[2, 1] - r[1, 2]) / (4 * q0)
    q2 = (r[0, 2] - r[2, 0]) / (4 * q0)
    q3 = (r[1, 0] - r[0, 1]) / (4 * q0)
    return [q0, q1, q2, q3]

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

def check_simple_facing(MODEL_DATA, cam_pos, blob_ids, angle_spec=90.0):
    results = {}
    for blob_id in blob_ids:
        blob_pos = MODEL_DATA[blob_id]
        o_to_led = blob_pos - [0, 0, 0]
        led_to_cam = blob_pos - cam_pos
        normalize = led_to_cam / np.linalg.norm(led_to_cam)
        facing_dot = np.dot(normalize, o_to_led)
        angle = np.radians(180.0 - angle_spec)
        if facing_dot < np.cos(angle):
            results[blob_id] = True
        else:
            results[blob_id] = False
        # print('blob_id ', blob_id, 'facing_dot ',  facing_dot, ' rad ', np.cos(angle))
    return results


def check_angle_and_facing(MODEL_DATA, DIRECTION, cam_pos, cam_dir, blob_ids, threshold_angle=80.0):
    results = {}

    cam_pose = {'position': vector3(*cam_pos), 'orient': quat(*cam_dir)}
    for blob_id in blob_ids:
        # Blob의 위치
        blob_pos = MODEL_DATA[blob_id]        
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
        # print(f"{blob_id} : facing_dot: {facing_dot} rad{rad}")
    return results
def reprojection_error(points3D, points2D, rvec, tvec, camera_k, dist_coeff):        
        points2D_reprojection, _ = cv2.projectPoints(points3D, np.array(rvec), np.array(tvec), camera_k, dist_coeff)
        # Squeeze the points2D_reprojection to match the dimensionality of points2D
        points2D_reprojection = points2D_reprojection.squeeze()
        RER = np.average(np.linalg.norm(points2D - points2D_reprojection, axis=1))
        # print('points2D:\n', points2D)
        # print('points2D_reprojection:\n', points2D_reprojection)
        # print('RER:', RER)
        return RER, points2D_reprojection
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




































    
'''
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
def recover_pose_essential_test_two(script_dir):
    def recover_3d_points_combinations(cam_info, K):
        recovered_3d_points = {}
        frame_keys = list(cam_info.keys())
        frame_pairs = list(combinations(frame_keys, 2))  # Generate combinations of frame pairs

        for frame_pair in frame_pairs:
            frame1, frame2 = frame_pair
            points1_2d = np.array(cam_info[frame1]["points2D_U"]['greysum'])
            points2_2d = np.array(cam_info[frame2]["points2D_U"]['greysum'])

            print('frame1, frame2:', frame1, frame2)
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
            print('t: ', t)
            # Scale the translation vector
            actual_movement = 0.01  # Assuming 0.01m movement in y-axis
            t = t * (actual_movement / np.linalg.norm(t))

            R1 = np.eye(3)
            t1 = np.zeros((3, 1))
            R2 = R
            t2 = t
            points_3d_homogeneous = cv2.triangulatePoints(np.hstack((R1, t1)), np.hstack((R2, t2)), points1_2d.T, points2_2d.T)
            points_3d = points_3d_homogeneous[:3, :] / points_3d_homogeneous[3, :]
            points_3d = points_3d * (actual_movement / np.linalg.norm(t))
            combination_key = (frame_pair, (frame1, frame2))
            recovered_3d_points[combination_key] = points_3d.T
            print('points_3d.T:',points_3d.T)

        return recovered_3d_points
 
    image_files = sorted(glob.glob(os.path.join(script_dir, f"./tmp/render/ESSENTIAL/" + '*.png')))
    frame_cnt = 0
    while True:
        print('\n')
        print(f"########## Frame {frame_cnt} ##########")

        # BLENDER와 확인해 보니 마지막 카메라 위치가 시작지점으로 돌아와서 추후 remake 3D 에서 이상치 발생 ( -1 )  
        if frame_cnt >= len(image_files):
            break
        frame_0 = cv2.imread(image_files[frame_cnt])
        filename = f"IMAGE Mode {os.path.basename(image_files[frame_cnt])}"
        if frame_0 is None or frame_0.size == 0:
            print(f"Failed to load {image_files[frame_cnt]}, frame_cnt:{frame_cnt}")
            continue

        draw_frame = frame_0.copy()
        _, frame_0 = cv2.threshold(cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                   cv2.THRESH_TOZERO)

        cv2.putText(draw_frame, f"frame_cnt {frame_cnt} [{filename}]", (draw_frame.shape[1] - 400, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        height, width = frame_0.shape
        center_x, center_y = width // 2, height // 2
        cv2.line(draw_frame, (0, center_y), (width, center_y), (255, 255, 255), 1)
        cv2.line(draw_frame, (center_x, 0), (center_x, height), (255, 255, 255), 1)
                # find Blob area by findContours
        blob_area = detect_led_lights(frame_0, 2, 5, 500)
        blob_cent = []
        points2D_D = []
        points2D_U = []
        LED_NUMBER = []
        for blob_id, bbox in enumerate(blob_area):
            (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            gcx, gcy, gsize = find_center(frame_0, (x, y, w, h))
            if gsize < BLOB_SIZE:
                continue

            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
            cv2.putText(draw_frame, f"{blob_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # cv2.putText(draw_frame, f"{int(gcx)},{int(gcy)},{gsize}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
            #             (255, 255, 255), 1)
            # blob_centers.append((gcx, gcy, bbox))
            # print(f"{blob_id} : {gcx}, {gcy}")
            LED_NUMBER.append(blob_id)
            temp_points_2d_d = np.array([gcx, gcy], dtype=np.float64)
            temp_points_2d_u = np.array(cv2.undistortPoints(temp_points_2d_d, camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][1])).reshape(-1, 2)
            points2D_D.append(temp_points_2d_d)
            points2D_U.append(temp_points_2d_u)
            


        CAMERA_INFO[f"{frame_cnt}"] = copy.deepcopy(CAMERA_INFO_STRUCTURE)
        points2D_D = np.array(np.array(points2D_D).reshape(len(points2D_D), -1), dtype=np.float64)
        points2D_U = np.array(np.array(points2D_U).reshape(len(points2D_U), -1), dtype=np.float64)
        CAMERA_INFO[f"{frame_cnt}"]['points2D']['greysum'] = points2D_D
        CAMERA_INFO[f"{frame_cnt}"]['points2D_U']['greysum'] = points2D_U            
        CAMERA_INFO[f"{frame_cnt}"]['LED_NUMBER'] =LED_NUMBER

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
            print('go next IMAGE')
        elif key == ord('b'):
            frame_cnt -= 1
            print('go back IMAGE')
    
    cv2.destroyAllWindows()

    for frame_cnt, cam_data in CAMERA_INFO.items():
        led_numbers = cam_data['LED_NUMBER']
        points2D = cam_data['points2D']['greysum']
        points2D_U = cam_data['points2D_U']['greysum']
        print('led_numbers:', led_numbers)
        print(points2D_U)

    recover_points = recover_3d_points_combinations(CAMERA_INFO, camera_matrix[0][0])
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
    
'''



def terminal_cmd(cmd_m, cmd_s):
    print('start ', terminal_cmd.__name__)
    try:
        result = subprocess.run([cmd_m, cmd_s], stdout=subprocess.PIPE).stdout.decode('utf-8')

        device_re = re.compile(b"Bus\s+(?P<bus>\d+)\s+Device\s+(?P<device>\d+).+ID\s(?P<id>\w+:\w+)\s(?P<tag>.+)$",
                               re.I)
        df = subprocess.check_output("lsusb")
        devices = []
        for i in df.split(b'\n'):
            if i:
                info = device_re.match(i)
                if info:
                    dinfo = info.groupdict()
                    dinfo['device'] = '/dev/bus/usb/%s/%s' % (dinfo.pop('bus'), dinfo.pop('device'))
                    devices.append(dinfo)
    except:
        print('exception')
        traceback.print_exc()
    else:
        print('done')
    finally:
        print(devices)
    temp = result.split('\n\n')
    Rift_Sensor = "Rift Sensor"
    print("==================================================")
    ret_val = []
    for i in range(len(temp)):
        if Rift_Sensor in temp[i]:
            ret_val.append(temp[i])
            print("add list rift_sensor", temp[i])
        else:
            print("skipping camera", temp[i])
    print("==================================================")
    return ret_val

def init_model_json(cam_dev_list):
    print('start ', init_model_json.__name__)
    camera_info_array = []
    try:
        for i in range(len(cam_dev_list)):
            cam_info = cam_dev_list[i].split('\n\t')
            camera_info_array.append({'name': cam_info[0], 'port': cam_info[1]})
    except:
        print('exception')
        traceback.print_exc()
    finally:
        print('done')
    return camera_info_array


def world_location_rotation_from_opencv(rvec, tvec, isCamera=True):
    R_BlenderView_to_OpenCVView = np.matrix([
        [1 if isCamera else -1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ])
    # Convert rvec to rotation matrix
    R_OpenCV, _ = cv2.Rodrigues(rvec)

    # Convert OpenCV R|T to Blender R|T
    R_BlenderView = R_BlenderView_to_OpenCVView @ np.matrix(R_OpenCV.tolist())
    T_BlenderView = R_BlenderView_to_OpenCVView @ vector3(tvec[0], tvec[1], tvec[2])

    print('tvec', tvec)
    # Invert rotation matrix
    R_BlenderView_inv = np.array(R_BlenderView).T
    print(R_BlenderView_inv)
    print(T_BlenderView)
    # Calculate location
    location = -1.0 * R_BlenderView_inv @ T_BlenderView
    # Convert rotation matrix to quaternion
    rotation = R_BlenderView_inv.to_quaternion()
    return location, rotation


def area_filter(x, y, POS):
    # print(f"{x} {y} {POS}")
    MODE = POS['mode']
    if MODE == CIRCLE:
        dx = np.abs(x - POS['circle'][0])
        dy = np.abs(y - POS['circle'][1])
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance <= POS['circle'][2]:
            return True
        else:
            return False
    else:
        sx = min(POS['rectangle'][0], POS['rectangle'][2])
        sy = min(POS['rectangle'][1], POS['rectangle'][3])
        w = np.abs(POS['rectangle'][0] - POS['rectangle'][2])
        h = np.abs(POS['rectangle'][1] - POS['rectangle'][3])
        # print(f"{sx} {sy} {w} {h}")
        if x >= sx and x <= sx + w and y >= sy and y <= sy + h:
            return True
        else:
            return False

import numpy as np
from scipy.spatial.distance import pdist

def fit_circle_2d(x, y, w=[], min_radius=0):
    # Existing code
    A = np.array([x, y, np.ones(len(x))]).T
    b = x**2 + y**2
    
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W, A)
        b = np.dot(W, b)
    
    c = np.linalg.lstsq(A, b, rcond=None)[0]

    # Existing code end

    xc = c[0]/2
    yc = c[1]/2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    
    # If radius is less than the minimum radius, adjust it to the minimum radius
    if r < min_radius:
        r = min_radius

    return xc, yc, r

from scipy.spatial.distance import pdist
def fit_circle_2d_fixed_center(x, y, center, w=[]):
    # Use the provided center
    xc, yc = center

    # Calculate radius based on the provided center
    r = np.mean(np.sqrt((x - xc)**2 + (y - yc)**2))

    # If provided weights, adjust the radius accordingly
    if len(w) == len(x):
        r = np.sum(w * np.sqrt((x - xc)**2 + (y - yc)**2)) / np.sum(w)

    return xc, yc, r