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
from essential_main import *

CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
angle_spec = 80
CV_PI = 3.14159


def camera_displacement(r1, r2, t1, t2):
    print('r1 ', r1)
    print('r2 ', r2)
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


def inverse_matrix(r12, t12, r1, t1):
    print('r12\n', r12)
    print('t12\n', t12)
    print('r1\n', r1)
    print('t1\n', t1)
    Rod1, _ = cv2.Rodrigues(r1)
    R1to2, _ = cv2.Rodrigues(r12)
    Rod2 = R1to2.dot(np.linalg.inv(Rod1.T))
    r2, _ = cv2.Rodrigues(Rod2)
    t2 = t12 + R1to2.dot(t1)

    print('r2 ', r2)
    print('t2 ', t2)
    return r2, t2.reshape(3, 1)


def make_camera_array():
    dist_to_controller = 0.5

    depth = 0.5
    step = 60
    dx = 360
    dz = 180

    cam_id = 0
    cam_pose = []
    for k in range(int(dx / step)):
        # for k in range(1):
        delta = math.radians(k * step)  # alpha : x 축과 이루는 각도 (0 ~ 360도)
        for m in range(int(dz / step)):
            theta = math.radians(m * step)  # beta : z 축과 이루는 각도 (0 ~ 180도)
            # theta = math.radians(90)  # beta : z 축과 이루는 각도 (0 ~ 180도)

            x = math.cos(delta) * math.sin(theta) * depth
            y = math.sin(delta) * math.sin(theta) * depth
            z = math.cos(theta) * depth
            # print(np.array([x, y, z]))

            u = -x / depth
            v = -y / depth
            w = -z / depth
            Z3, Z2, Z1 = w, v, u
            if Z3 == -1 or Z3 == 1:
                alpha = 0
            else:
                # alpha = math.degrees(math.acos(np.clip(-Z2 / math.sqrt(1 - Z3*Z3), -1.0, 1.0)))
                alpha = math.degrees(math.acos(np.clip(-Z2 / math.sqrt(1 - Z3 * Z3), -1.0, 1.0)))
            if Z1 < 0:
                alpha = 360 - alpha
            beta = math.degrees(math.acos(Z3))
            # beta = math.degrees(math.acos(Z3))
            gamma = 0

            cam_pose.append({
                'idx': cam_id,
                'position_view': vector3(x, y, z),
                'position': vector3(0.0, 0.0, dist_to_controller),
                # 'direction' : ('xyz', [-(x),-(y),-(z)]),
                # 'orient': quat(-(x),-(y),-(z))
                'direction': ('zxz', [-alpha, -beta, -gamma]),
                'orient': get_quat_from_euler('zxz', [-alpha, -beta, -gamma])
                # 'orient': quat(0,0,0)
            })
            cam_id += 1
    # check_projection(cam_pose, cam_array)

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

    MAX_DEGREE = 60
    cam_id = 0
    cam_pose = []
    # 30cm
    # for dist in range(1):
    #     for idx in range(MAX_DEGREE):
    #         degree = idx * 3
    #         cam_pose.append({
    #             'idx': cam_id,
    #             'position': vector3(0.0, 0.0, 0.5 + (dist * 0.1)),
    #             'orient': [math.radians(0), math.radians(55), math.radians(degree)]
    #         })
    #         cam_pose.append({
    #             'idx': cam_id + 1,
    #             'position': vector3(0.0, 0.0, 0.5 + (dist * 0.1)),
    #             'orient': [math.radians(0), math.radians(55), math.radians(-(180 - degree))]
    #         })
    #         cam_id += 2

    # cam_pose.append({
    #     'idx': cam_id,
    #     'position': vector3(0.0, 0.0, 0.5),
    #     'orient': get_quat_from_euler('zxy', [0, 55, 10])
    # })
    # cam_pose.append({
    #     'idx': cam_id + 1,
    #     'position': vector3(0.0, 0.0, 0.5),
    #     'orient': get_quat_from_euler('zxy', [0, 55, 15])
    # })

    # ToDo
    # 여기 study 필요
    # 정답이 안나오는 이상한 위치가 있음
    cam_pose.append({
        'idx': 0,
        'position': vector3(0.0, 0.0, 0.5),
        'orient': [5.0 * CV_PI / 180.0, -3.0 * CV_PI / 180.0, 8.0 * CV_PI / 180.0]
    })
    cam_pose.append({
        'idx': 1,
        'position': vector3(0.0, 0.0, 0.5),
        'orient': [-3.0 * CV_PI / 180.0, -5.0 * CV_PI / 180.0, 1.0 * CV_PI / 180.0]
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
    rvecs1 = np.array(cam_pose[0]['orient'])


    # 여기서 오리지널 pts를 투영 시켜 본다.
    print('rvecs1\n', np.array(cam_pose[0]['orient']))
    print('tvecs1\n', np.array(cam_pose[0]['position']))
    imagePoint1 = []
    # Planar structure
    # DistCoeff 정보 넣지 않음
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

    if DO_UNDISTORT == ENABLE:
        list_2d_undistorted_o1 = cv2.undistortPoints(imagePoint1,
                                                     temp_camera_k,
                                                     None)
        list_2d_undistorted_o1 = np.reshape(list_2d_undistorted_o1, (1, 6, 2)).copy()

        list_2d_undistorted_o2 = cv2.undistortPoints(imagePoint2,
                                                     temp_camera_k,
                                                     None)
        list_2d_undistorted_o2 = np.reshape(list_2d_undistorted_o2, (1, 6, 2)).copy()


    # print imagePoint result
    # print('imagePoints1\n', imagePoint1)
    # print('imagePoints2\n', imagePoint2)
    rvec1to2, tvec1to2 = camera_displacement(rvecs1, rvecs2, np.array(cam_pose[0]['position']), np.array(cam_pose[1]['position']))

    # 펀더멘털 매트릭스 왜 안나옴????
    # 8개는 되야 제대로 나옴.....ㅅㅂ
    # F, mask = cv2.findFundamentalMat(imagePoint1, imagePoint2, method=cv2.FM_RANSAC)
    # print('F\n', F)
    E, mask = cv2.findEssentialMat(list_2d_undistorted_o1, list_2d_undistorted_o2, 1.0, (0, 0), cv2.RANSAC, 0.999, 1, None)
    # F2 = np.linalg.inv(temp_camera_k.T).dot(E).dot(np.linalg.inv(temp_camera_k))
    # E2 = temp_camera_k.T.dot(F2).dot(temp_camera_k)
    # print('F2\n', F2)
    # print('E\n', E)
    # print('E2\n', E2)

    _, Rvecs, Tvecs, M = cv2.recoverPose(E, list_2d_undistorted_o1, list_2d_undistorted_o2)
    recover_rvec, _ = cv2.Rodrigues(Rvecs)
    print('After Recover Pose')
    print('Rvecs\n', recover_rvec.T)
    print('Tvecs\n', Tvecs.T)
    scalePose = tvec1to2[2] / Tvecs[2][0]
    print('scalePose : ', scalePose)

    new_Tvecs = scalePose*Tvecs
    print('new_Tvecs\n', new_Tvecs.T)

    inverse_matrix(rvec1to2, tvec1to2, rvecs1, np.array(cam_pose[0]['position']))

    # trianglutatePoints
    print('origin\n', origin_leds)
    print('solvePnP + trianglutePoint')

    if DO_UNDISTORT == ENABLE:
        _, r_1, t_1, inliers = cv2.solvePnPRansac(origin_leds, list_2d_undistorted_o1,
                                                  cameraK,
                                                  distCoeff)
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

    print('r_1\n', r_1.T)
    print('t_1\n', t_1.T)
    print('r_2\n', r_2)
    print('t_2\n', t_2)

    print(camera_displacement(r_1, r_2, t_1, t_2))
    inverse_matrix(rvec1to2, tvec1to2, r_1.T[0], t_1.T[0])

    left_rotation, jacobian = cv2.Rodrigues(r_1)
    right_rotation, jacobian = cv2.Rodrigues(r_2)
    left_projection = np.hstack((left_rotation, t_1))
    # print('left_project\n', left_projection)
    right_projection = np.hstack((right_rotation, t_2))
    # print('right_project\n', right_projection)
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
    # pm1 = np.zeros((3, 4))
    pm1 = np.hstack((np.eye(3), np.zeros((3, 1))))

    d_r, _ = cv2.Rodrigues(np.eye(3))
    d_t = np.zeros((3, 1))

    # print('d_r ', d_r, ' d_t ', d_t)
    # 결국 여기서 default matrix와 recoverPose로 구한 R|T 변위가 위 테스트 카메라 위치 변위와 같아야 함
    rvec_1to2, tvec_1to2 = camera_displacement(d_r, recover_rvec, d_t, new_Tvecs)
    # 여기서도 물리적인(?) primary camera R|T를 알아야 한다. 왜냐면 Essential Matrix는 카메라 월드의 좌표 반영이기 때문에
    # 어떻게 보는지 모른다. -> 좌표 돈다.
    # Primary Camera R|T에 변위를 inverse하여 R2 , T2를 추정한다.
    inv_rvec, inv_tvec = inverse_matrix(rvec_1to2, tvec_1to2.T[0], r_1.T[0], t_1.T[0])

    print('inv_rvec\n', inv_rvec)
    print('inv_tvec\n', inv_tvec)

    # P = np.hstack((Rvecs, new_Tvecs))
    # P = np.hstack((inv_rvec, inv_tvec))
    # print('pm1\n', pm1)
    # print('P\n', P)

    inv_right_rotation, jacobian = cv2.Rodrigues(inv_rvec)
    inv_right_projection = np.hstack((inv_right_rotation, inv_tvec))
    print('int_right_projection\n', inv_right_projection)

    triangulation_r = cv2.triangulatePoints(left_projection,  inv_right_projection,
                                            list_2d_undistorted_o1,
                                            list_2d_undistorted_o2)
    homog_points_r = triangulation_r.transpose()
    get_points = cv2.convertPointsFromHomogeneous(homog_points_r)

    # get_points = compute3Dpoints(pm1, P, list_2d_undistorted_o1, list_2d_undistorted_o2)
    print('get_points(recoverPose)\n', get_points)


def compute3Dpoints(P1, P2, npts1, npts2):
    # computes object point coordinates from photo points correspondence using DLT method
    ptsh3d = cv2.triangulatePoints(P1, P2, npts1.T, npts2.T).T

    pts_sps = copy.deepcopy(ptsh3d[:, :3])
    for i, pt in enumerate(ptsh3d):
        pt = (pt / pt[3])[:3]
        pts_sps[i, :] = pt[:3]

    return pts_sps
