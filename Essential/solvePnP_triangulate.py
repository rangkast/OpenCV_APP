import copy
import itertools
import json
import math
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import re

from definition import *
from ransac_test import *
from essential_test import *

CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
angle_spec = 80
CV_PI = 3.14159


def recover_test():
    return


def make_camera_array_2():
    MAX_DEGREE = 60
    cam_id = 0
    cam_pose = []
    # 30cm
    for dist in range(1):
        for idx in range(MAX_DEGREE):
            degree = idx * 3
            cam_pose.append({
                'idx': cam_id,
                'position': vector3(0.0, 0.0, 0.5 + (dist * 0.1)),
                'orient': get_quat_from_euler('zxy', [0, 55, degree])
            })
            cam_pose.append({
                'idx': cam_id + 1,
                'position': vector3(0.0, 0.0, 0.5 + (dist * 0.1)),
                'orient': get_quat_from_euler('zxy', [0, 55, -(180 - degree)])
            })
            cam_id += 2

    return cam_pose


if __name__ == "__main__":
    print('start ransac test main')
    plt.style.use('default')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    origin = 'rift_6'
    target = 'rift_2'
    # read origin data
    read_led_pts(origin)
    # read target data
    read_led_pts(target)

    print('origin')
    for led in leds_data[origin]:
        print(led)
    print('target')
    for led in leds_data[target]:
        print(led)

    camera_array_origin = init_model_json("../Calibration/jsons/")

    cam_pose = []
    cam_pose.append({
        'idx': 0,
        'position': vector3(0.0, 0.0, 0.5),
        'orient': get_quat_from_euler('zxy', [0, 55, 10])
    })
    cam_pose.append({
        'idx': 1,
        'position': vector3(0.0, 0.0, 0.5),
        'orient': get_quat_from_euler('zxy', [0, 55, 15])
    })

    # Test Code
    leds_array = [0, 1, 2, 3, 4, 5]
    temp_camera_k = camera_array_origin[0]['cam_cal']['cameraK']
    temp_dist_coeff = camera_array_origin[0]['cam_cal']['dist_coeff']

    # leds objectPoints
    origin_leds = []
    for led_num in leds_array:
        origin_leds.append([leds_data[origin][led_num]['pos']])
    origin_leds = np.array(origin_leds, dtype=np.float64)

    # 여기서 오리지널 pts를 투영 시켜 본다.
    obj_cam_pos_n = np.array(cam_pose[0]['position'])
    rotR = R.from_quat(np.array(cam_pose[0]['orient']))
    Rod, _ = cv2.Rodrigues(rotR.as_matrix())

    imagePoint1 = []
    oret = cv2.projectPoints(origin_leds, Rod, obj_cam_pos_n,
                             temp_camera_k,
                             None)
    o_xx, o_yy = oret[0].reshape(len(origin_leds), 2).transpose()
    for i in range(len(o_xx)):
        imagePoint1.append([[o_xx[i], o_yy[i]]])
    imagePoint1 = np.array(imagePoint1, dtype=np.float64)
    imagePoint1 = np.reshape(imagePoint1, (1, 6, 2)).copy()

    print('imagePoint1\n', imagePoint1)

    obj_cam_pos_n = np.array(cam_pose[1]['position'])
    rotR = R.from_quat(np.array(cam_pose[1]['orient']))
    Rod, _ = cv2.Rodrigues(rotR.as_matrix())

    imagePoint2 = []
    oret = cv2.projectPoints(origin_leds, Rod, obj_cam_pos_n,
                             temp_camera_k,
                             None)
    o_xx, o_yy = oret[0].reshape(len(origin_leds), 2).transpose()
    for i in range(len(o_xx)):
        imagePoint2.append([[o_xx[i], o_yy[i]]])
    imagePoint2 = np.array(imagePoint2, dtype=np.float64)
    imagePoint2 = np.reshape(imagePoint2, (1, 6, 2)).copy()
    print('imagePoint2\n', imagePoint2)

    # trianglutatePoints
    print('origin\n', origin_leds)
    print('solvePnP + trianglutePoint')

    if DO_UNDISTORT == ENABLE:
        list_2d_undistorted_o1 = cv2.undistortPoints(imagePoint1,
                                                     temp_camera_k,
                                                     None)
        list_2d_undistorted_o1 = np.reshape(list_2d_undistorted_o1, (1, 6, 2)).copy()

        _, r_1, t_1, inliers = cv2.solvePnPRansac(origin_leds, list_2d_undistorted_o1,
                                                  cameraK,
                                                  distCoeff)

        list_2d_undistorted_o2 = cv2.undistortPoints(imagePoint2,
                                                     temp_camera_k,
                                                     None)
        list_2d_undistorted_o2 = np.reshape(list_2d_undistorted_o2, (1, 6, 2)).copy()

        _, r_2, t_2, inliers = cv2.solvePnPRansac(origin_leds, list_2d_undistorted_o2,
                                                  cameraK,
                                                  distCoeff)
    else:
        _, r_1, t_1, inliers = cv2.solvePnPRansac(origin_leds, imagePoint1,
                                                  temp_camera_k,
                                                  None)

        _, r_2, t_2, inliers = cv2.solvePnPRansac(origin_leds, imagePoint2,
                                                  temp_camera_k,
                                                  None)

    # ToDo
    # SolvePnP + triangluatePoints
    left_rotation, jacobian = cv2.Rodrigues(r_1)
    right_rotation, jacobian = cv2.Rodrigues(r_2)

    # RT = np.zeros((3, 4))
    # RT[:3, :3] = left_rotation
    # RT[:3, 3] = prime_T.transpose()
    # left_projection = np.dot(cameraK, RT)
    left_projection = np.hstack((left_rotation, t_1))
    print('left_project\n', left_projection)

    # RT = np.zeros((3, 4))
    # RT[:3, :3] = right_rotation
    # RT[:3, 3] = t_o.transpose()
    # right_projection = np.dot(cameraK, RT)
    right_projection = np.hstack((right_rotation, t_2))
    print('right_project\n', right_projection)

    if DO_UNDISTORT == ENABLE:
        triangulation = cv2.triangulatePoints(left_projection, right_projection,
                                          list_2d_undistorted_o1,
                                          list_2d_undistorted_o2)
    else:
        triangulation = cv2.triangulatePoints(left_projection, right_projection,
                                          imagePoint1,
                                          imagePoint2)

    homog_points = triangulation.transpose()

    sget_points = cv2.convertPointsFromHomogeneous(homog_points)

    print('get_points(solvePnP)\n', sget_points)
