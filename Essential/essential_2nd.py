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


def camera_displacement(r1, r2, t1, t2):
    Rod1, _ = cv2.Rodrigues(r1)
    Rod2, _ = cv2.Rodrigues(r2)
    R1to2 = Rod2.dot(Rod1.T)
    rvec1to2, _ = cv2.Rodrigues(R1to2)
    tvec1to2 = -R1to2.dot(t1) + t2

    print('Rod1\n', Rod1)
    print('Rod2\n', Rod2)
    print('rvec1to2\n', rvec1to2.T)
    print('tvec1to2\n', tvec1to2.T)

    return rvec1to2, tvec1to2

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
    #
    # cam_pose.append({
    #     'idx': 0,
    #     'position': vector3(0.0, 0.0, 0.5),
    #     'orient': get_quat_from_euler('zxy', [0, 55, 10])
    # })
    # cam_pose.append({
    #     'idx': 1,
    #     'position': vector3(0.0, 0.0, 0.5),
    #     'orient': get_quat_from_euler('zxy', [0, 55, 15])
    # })

    cam_pose.append({
        'idx': 0,
        'position': vector3(0.0, 0.0, 0.5),
        'orient': [5.0 * CV_PI / 180.0, -3.0 * CV_PI / 180.0, 8.0 * CV_PI / 180.0]
    })
    cam_pose.append({
        'idx': 1,
        'position': vector3(0.0, 0.0, 0.5),
        'orient': [-3.0 * CV_PI / 180.0, -5.0 * CV_PI / 180.0, -3.0 * CV_PI / 180.0]
    })

    #
    # for i, data in enumerate(cam_pose):
    #     pts_facing, leds_array = get_facing_dot(origin, data, 0)
    #     print('f: ', leds_array)

    # Test Code
    leds_array = [0, 1, 2, 3, 4, 5]
    temp_camera_k = camera_array_origin[0]['cam_cal']['cameraK']
    temp_dist_coeff = camera_array_origin[0]['cam_cal']['dist_coeff']

    # leds objectPoints
    origin_leds = []
    for led_num in leds_array:
        origin_leds.append([leds_data[origin][led_num]['pos']])
    origin_leds = np.array(origin_leds, dtype=np.float64)
    rvecs1 = np.array(cam_pose[0]['orient'])
    print('rvecs1\n', np.array(cam_pose[1]['orient']).T)

    # 여기서 오리지널 pts를 투영 시켜 본다.
    # Rod1, _ = cv2.Rodrigues(R.from_quat(np.array(cam_pose[0]['orient'])).as_matrix())

    print('rvecs1\n', np.array(cam_pose[0]['orient']))
    print('tvecs1\n', np.array(cam_pose[0]['position']))
    imagePoint1 = []
    oret = cv2.projectPoints(origin_leds, rvecs1, np.array(cam_pose[0]['position']),
                             temp_camera_k,
                             None)
    o_xx, o_yy = oret[0].reshape(len(origin_leds), 2).transpose()
    for i in range(len(o_xx)):
        imagePoint1.append([[o_xx[i], o_yy[i]]])
    imagePoint1 = np.array(imagePoint1, dtype=np.float64)
    imagePoint1 = np.reshape(imagePoint1, (1, 6, 2)).copy()
    rvecs2 = np.array(cam_pose[1]['orient'])
    # Rod2, _ = cv2.Rodrigues(R.from_quat(np.array(cam_pose[1]['orient'])).as_matrix())
    print('rvecs2\n', np.array(cam_pose[1]['orient']))
    print('tvecs2\n', np.array(cam_pose[1]['position']))
    imagePoint2 = []
    oret = cv2.projectPoints(origin_leds, rvecs2, np.array(cam_pose[1]['position']),
                             temp_camera_k,
                             None)
    o_xx, o_yy = oret[0].reshape(len(origin_leds), 2).transpose()
    for i in range(len(o_xx)):
        imagePoint2.append([[o_xx[i], o_yy[i]]])
    imagePoint2 = np.array(imagePoint2, dtype=np.float64)
    imagePoint2 = np.reshape(imagePoint2, (1, 6, 2)).copy()

    # print imagePoint result
    print('imagePoints1\n', imagePoint1)
    print('imagePoints2\n', imagePoint2)
    rvec1to2, tvec1to2 = camera_displacement(rvecs1, rvecs2, np.array(cam_pose[0]['position']), np.array(cam_pose[1]['position']))
    # 펀더멘털 매트릭스 왜 안나옴????
    # 8개는 되야 제대로 나옴.....ㅅㅂ
    F, mask = cv2.findFundamentalMat(imagePoint1, imagePoint2, method=cv2.FM_RANSAC)
    print('F\n', F)
    E, mask = cv2.findEssentialMat(imagePoint1, imagePoint2, temp_camera_k, cv2.RANSAC, 0.999, 1, None)
    F2 = np.linalg.inv(temp_camera_k.T).dot(E).dot(np.linalg.inv(temp_camera_k))

    E2 = temp_camera_k.T.dot(F2).dot(temp_camera_k)
    print('F2\n', F2)
    print('E\n', E)
    print('E2\n', E2)

    _, Rvecs, Tvecs, M = cv2.recoverPose(E, imagePoint1, imagePoint2)
    recover_rvec, _ = cv2.Rodrigues(Rvecs)
    print('After Recover Pose\n')
    print('Rvecs\n', recover_rvec)
    print('Tvecs\n', Tvecs)
    scalePose = tvec1to2[2] / Tvecs[2][0]
    print('scalePose : ', scalePose)

    new_Tvecs = scalePose*Tvecs
    print('new_Tvecs\n', new_Tvecs)
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

    print('r_1\n', r_1)
    print('t_1\n', t_1)

    print(camera_displacement(r_1, r_2, t_1, t_2))

    left_rotation, jacobian = cv2.Rodrigues(r_1)
    right_rotation, jacobian = cv2.Rodrigues(r_2)
    left_projection = np.hstack((left_rotation, t_1))
    print('left_project\n', left_projection)
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

    print('recoverPose + triangluatePoint')
    P = np.hstack((Rvecs, new_Tvecs))
    # pm1 = np.zeros((3, 4))
    pm1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    print('pm1\n', pm1)
    print('P\n', P)
    d_r, _ = cv2.Rodrigues(np.eye(3))
    d_t, _ = cv2.Rodrigues(np.eye(3))

    print('d_r ', d_r, ' d_t ', d_t)
    print(camera_displacement(d_r, recover_rvec, d_t, new_Tvecs))

    # left_rotation, jacobian = cv2.Rodrigues(d_r)
    # left_projection = np.hstack((left_rotation, d_t))
    # print('left_project\n', left_projection)
    #
    # right_rotation, jacobian = cv2.Rodrigues(recover_rvec)
    # right_projection = np.hstack((right_rotation, new_Tvecs))
    # print('right_project\n', right_projection)

    triangulation_r = cv2.triangulatePoints(pm1, P,
                                            list_2d_undistorted_o1,
                                            list_2d_undistorted_o2)
    homog_points_r = triangulation_r.transpose()
    get_points = cv2.convertPointsFromHomogeneous(homog_points_r)
    print('get_points(recoverPose)\n', get_points)
