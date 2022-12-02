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

CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
angle_spec = 80


def make_camera_array():
    dist_to_controller = 0.5

    depth = 0.5
    step = 30
    dx = 360
    dz = 180

    cam_id = 0
    cam_pose = []

    for k in range(int(dx / step)):
        # for k in range(1):
        delta = math.radians(k * step)  # alpha : x 축과 이루는 각도 (0 ~ 360도)
        for m in range(int(dz / step)):
            theta = math.radians(m * step)  # beta : z 축과 이루는 각도 (0 ~ 180도)

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


def check_facing_dot(target, cam_pose):
    pts_facing = []
    leds_array = []
    for data in target:
        idx = int(data['idx'])
        temp = transfer_point(vector3(data['pos'][0], data['pos'][1], data['pos'][2]), cam_pose)
        ori = rotate_point(vector3(data['dir'][0], data['dir'][1], data['dir'][2]), cam_pose)
        normal = nomalize_point(vector3(temp.x, temp.y, temp.z))
        facing_dot = get_dot_point(normal, ori)
        angle = math.radians(180.0 - angle_spec)
        rad = np.cos(angle)
        # rad  -0.3420201433256687
        if facing_dot < rad:
            pts_facing.append({'idx': idx, 'pos': list(map(float, [data['pos'][0], data['pos'][1], data['pos'][2]])),
                               'dir': list(map(float, [data['dir'][0], data['dir'][1], data['dir'][2]])),
                               'angle': facing_dot})
            leds_array.append(idx)
    return pts_facing, leds_array


def get_facing_dot(fname, cam_pose, print_val):
    pts_facing = []
    leds_array = []

    if print_val == 1 and cam_pose['idx'] <= 1:
        print(f"[DEBUG_SM][{fname}] cam_id{cam_pose}")
    for i, data in enumerate(leds_data[fname]):
        # 카메라 pose 기준으로 led 좌표 변환 & led의 방향 벡터 변환
        led_id = int(data['idx'])
        # origin
        temp = transfer_point(vector3(data['pos'][0], data['pos'][1], data['pos'][2]), cam_pose)
        ori = rotate_point(vector3(data['dir'][0], data['dir'][1], data['dir'][2]), cam_pose)
        _pos_trans = list(map(float, [temp.x, temp.y, temp.z]))
        # 단위 벡터 생성
        normal = nomalize_point(vector3(temp.x, temp.y, temp.z))
        _dir_trans = list(map(float, [ori.x, ori.y, ori.z]))
        # facing dot 찾기/
        ori = nomalize_point(ori)
        facing_dot = get_dot_point(normal, ori)
        rad = np.arccos(np.clip(facing_dot, -1.0, 1.0))
        deg = np.rad2deg(rad)
        angle = math.radians(180.0 - angle_spec)
        rad = np.cos(angle)

        if facing_dot < rad:
            pts_facing.append({'idx': led_id, 'pos': list(map(float, [data['pos'][0], data['pos'][1], data['pos'][2]])),
                               'dir': list(map(float, [data['dir'][0], data['dir'][1], data['dir'][2]])),
                               'pattern': 0})

            leds_array.append(led_id)

    return pts_facing, leds_array


def color_loop(idx):
    if idx % 2 == 0:
        return 'red'
    else:
        return 'blue'


def check_projection(cam_pose, cam_array):
    plt.style.use('default')
    fig_ransac = plt.figure(figsize=(15, 15))
    plt.rc('xtick', labelsize=5)  # x축 눈금 폰트 크기
    plt.rc('ytick', labelsize=5)
    for cam_num, cam_pose_data in enumerate(cam_pose):

        # ax_ransac = fig_ransac.add_subplot()
        # ax_ransac.set_title('cam: 'f'{cam_num}')

        length = round((len(cam_pose) + 2) / 4)
        if length == 0:
            length = 1
        fig_ransac.tight_layout()
        ax_ransac = fig_ransac.add_subplot(4, length, cam_num + 1)
        ax_ransac.set_title('cam: 'f'{cam_num}')
        # o_pts_facing, leds_array = np.array(check_facing_dot(leds_data[origin], cam_pose_data))
        o_pts_facing, leds_array = get_facing_dot(origin, cam_pose_data, 0)

        print('f: ', leds_array)
        print('cam id: ', cam_pose_data['idx'], ' ori: ', cam_pose_data)
        # print('n ', new_pts_facing)
        # facing dot 과 좌표를 찾는다.

        origin_leds = []

        for x in o_pts_facing:
            origin_leds.append([[x['pos'][0], x['pos'][1], x['pos'][2]]])
        origin_leds = np.array(origin_leds, dtype=np.float64)

        # 여기서 오리지널 pts를 투영 시켜 본다.
        obj_cam_pos_n = np.array(cam_pose_data['position'])
        rotR = R.from_quat(np.array(cam_pose_data['orient']))
        Rod, _ = cv2.Rodrigues(rotR.as_matrix())

        cam_num = 0
        oret = cv2.projectPoints(origin_leds, Rod, obj_cam_pos_n,
                                 cam_array[cam_num]['cam_cal']['cameraK'],
                                 cam_array[cam_num]['cam_cal']['dist_coeff'])
        o_xx, o_yy = oret[0].reshape(len(origin_leds), 2).transpose()
        blobs_2d_origin = []
        for i in range(len(o_xx)):
            blobs_2d_origin.append([[o_xx[i], o_yy[i]]])

        print(blobs_2d_origin)
        #
        origin_pts = []
        for i in range(len(o_xx)):
            _opos = list(map(float, [o_xx[i], o_yy[i]]))
            origin_pts.append({'idx': leds_array[i], 'pos': _opos, 'reserved': 0})
        #
        color = color_loop(cam_num)
        draw_dots(2, origin_pts, ax_ransac, color)
        #
        max_x = 0
        max_y = 0
        if abs(max(o_xx)) > max_x: max_x = abs(max(o_xx))
        if abs(min(o_xx)) > max_x: max_x = abs(min(o_xx))
        if abs(max(o_yy)) > max_y: max_y = abs(max(o_yy))
        if abs(min(o_yy)) > max_y: max_y = abs(min(o_yy))
        dimen = max(max_x, max_y)
        dimen *= 1.1
        ax_ransac.set_xlim([0, 1280])
        ax_ransac.set_ylim([0, 960])
        #
        #
        # blobs_2d = np.array(blobs_2d_origin, dtype=np.float64)
        # blobs_2d = blobs_2d[:, 0]


def essential_test(cam_array, origin, target):
    cam_pose = make_camera_array()
    # for cam_array_data in cam_array:
    #     model = np.array(cam_array_data['model']['leds'])
    #
    #     temp_camera_k = cam_array_data['cam_cal']['cameraK']
    #     temp_dist_coeff = cam_array_data['cam_cal']['dist_coeff']
    #
    #     print('m: ', model)

    cam_pose_map = {}
    for i, data in enumerate(cam_pose):
        pts_facing, leds_array = get_facing_dot(origin, data, 0)
        # pts_facing, leds_array = np.array(check_facing_dot(leds_data[origin], data))
        cnt = len(pts_facing)

        if cnt < 6:
            continue

        # print(leds_array)
        # print('key values')
        keys = sliding_window(leds_array, 6)

        for key_data in keys:
            key_string = ','.join(str(e) for e in key_data)
            # print(key_string)

            if key_string in cam_pose_map:
                # print('found')
                cam_pose_map[key_string].append({'camera': data, 'leds': key_data})
            else:
                # print('not found')
                cam_pose_map[key_string] = [{'camera': data, 'leds': key_data}]
        # print('\n')

        new_pt = data['position_view']
        ax.scatter(new_pt.x, new_pt.y, new_pt.z, marker='.', color='red', s=20)
        label = (f"{data['idx']}")
        ax.text(new_pt.x, new_pt.y, new_pt.z, label, size=10)

    for i, key in enumerate(cam_pose_map):
        acc_cams_array = cam_pose_map.get(key)
        length = len(acc_cams_array)

        cam_num = 0
        temp_camera_k = cam_array[cam_num]['cam_cal']['cameraK']
        temp_dist_coeff = cam_array[cam_num]['cam_cal']['dist_coeff']
        pp = (temp_camera_k[0, 2], temp_camera_k[1, 2])
        focal = temp_camera_k[0, 0]
        print('focal ', focal, ' pp ', pp)

        if length > 1:
            print('group ', i, ' len ', length)
            print('leds ', key)
            prime_R = NOT_SET
            prime_T = NOT_SET
            prime_C_d = NOT_SET
            prime_C_u = NOT_SET
            for cnt, info in enumerate(acc_cams_array):
                print(info)

                origin_leds = []

                for led_num in info['leds']:
                    origin_leds.append([leds_data[origin][led_num]['pos']])
                origin_leds = np.array(origin_leds, dtype=np.float64)
                # print(origin_leds)
                # 여기서 오리지널 pts를 투영 시켜 본다.
                obj_cam_pos_n = np.array(info['camera']['position'])
                rotR = R.from_quat(np.array(info['camera']['orient']))
                Rod, _ = cv2.Rodrigues(rotR.as_matrix())

                oret = cv2.projectPoints(origin_leds, Rod, obj_cam_pos_n,
                                         temp_camera_k,
                                         temp_dist_coeff)
                o_xx, o_yy = oret[0].reshape(len(origin_leds), 2).transpose()
                blobs_2d_origin = []
                for i in range(len(o_xx)):
                    blobs_2d_origin.append([[o_xx[i], o_yy[i]]])

                # print(blobs_2d_origin)
                blobs_2d_distort_o = np.array(blobs_2d_origin, dtype=np.float64)
                blobs_2d_distort_o = np.reshape(blobs_2d_distort_o, (1, 6, 2)).copy()

                pts1 = np.ascontiguousarray(blobs_2d_distort_o, np.float32)

                list_2d_undistorted_o = cv2.undistortPoints(pts1,
                                                            temp_camera_k,
                                                            temp_dist_coeff)
                list_2d_undistorted_o = np.reshape(list_2d_undistorted_o, (1, 6, 2)).copy()

                _, r_o, t_o, inliers = cv2.solvePnPRansac(origin_leds, list_2d_undistorted_o,
                                                          cameraK,
                                                          distCoeff)
                if cnt == 0:
                    prime_R = r_o.copy()
                    prime_T = t_o.copy()

                    prime_C_d = blobs_2d_distort_o.copy()
                    prime_C_u = list_2d_undistorted_o.copy()

                    print('prime R\n', prime_R, '\nprime T\n', prime_T)
                    prime_cam_id = int(info['camera']['idx'])
                    prime_cam_info = info.copy()

                else:
                    # # Remove the fisheye distortion from the points
                    # pts0 = cv2.fisheye.undistortPoints(prime_C_d, temp_camera_k, temp_dist_coeff, P=temp_camera_k)
                    # pts2 = cv2.fisheye.undistortPoints(blobs_2d_distort_o, temp_camera_k, temp_dist_coeff, P=temp_camera_k)
                    # # pts0 = prime_C_u
                    # # pts2 = list_2d_undistorted_o
                    # # Keep only the points that make geometric sense
                    # # TODO: find a more efficient way to apply the mask
                    # E, mask = cv2.findEssentialMat(pts0, pts2, cameraK, cv2.RANSAC, 0.999, 1, None)
                    #
                    # # E, mask = cv2.findEssentialMat(prime_C_d, blobs_2d_distort_o,
                    # #                                threshold=0.05,
                    # #                                prob=0.95,
                    # #                                method=cv2.RANSAC,
                    # #                                focal=focal,
                    # #                                pp=pp)
                    #
                    # _, Rots, t, mask = cv2.recoverPose(E, pts0, pts2, cameraMatrix=cameraK, mask=mask)
                    # pts0_m = []
                    # pts2_m = []
                    # print('mask\n', mask, ' len ', len(mask))
                    # print('pts0\n', pts0)
                    # print('pts2\n', pts2)
                    # for ids in range(len(mask)):
                    #     if mask[ids] == 1:
                    #         pts0_m.append(pts0[:, ids])
                    #         pts2_m.append(pts2[:, ids])
                    # pts0 = np.array(pts0_m).T.reshape(2, -1)
                    # pts2 = np.array(pts2_m).T.reshape(2, -1)
                    #
                    # # Setup the projection matrices
                    # Rots = np.eye(3)
                    # t0 = np.array([[0], [0], [0]])
                    # t2 = np.array([[0], [0], [2]])
                    # P0 = np.dot(temp_camera_k, np.concatenate((Rots, t0), axis=1))
                    # P2 = np.dot(temp_camera_k, np.concatenate((Rots, t2), axis=1))
                    #
                    # # Find the keypoint world homogeneous coordinates assuming img0 is the world origin
                    # X = cv2.triangulatePoints(P0, P2, pts0, pts2)
                    # print('X\n', X)
                    #
                    # homog_points = X.transpose()
                    # get_points = cv2.convertPointsFromHomogeneous(homog_points)
                    # print('getpoints\n', get_points)
                    #
                    # # Convert from homogeneous cooridinates
                    # X /= X[3]
                    # objPts = X.T[:, :3]
                    #
                    # print('objPts\n', objPts)
                    # # Find the pose of the second frame
                    # _, rvec, tvec, inliers = cv2.solvePnPRansac(objPts, pts2.T, cameraK, None)
                    # print(rvec)
                    # print(tvec)

                    #
                    #

                    print('r_o\n', r_o)
                    print('t_o\n', t_o)

                    # ToDo
                    # SolvePnP + triangluatePoints
                    left_rotation, jacobian = cv2.Rodrigues(prime_R)
                    right_rotation, jacobian = cv2.Rodrigues(r_o)

                    # RT = np.zeros((3, 4))
                    # RT[:3, :3] = left_rotation
                    # RT[:3, 3] = prime_T.transpose()
                    # left_projection = np.dot(cameraK, RT)
                    left_projection = np.hstack((left_rotation, prime_T))
                    print('left_project\n', left_projection)

                    # RT = np.zeros((3, 4))
                    # RT[:3, :3] = right_rotation
                    # RT[:3, 3] = t_o.transpose()
                    # right_projection = np.dot(cameraK, RT)
                    right_projection = np.hstack((right_rotation, t_o))
                    print('right_project\n', right_projection)

                    triangulation = cv2.triangulatePoints(left_projection, right_projection,
                                                          prime_C_u,
                                                          list_2d_undistorted_o)
                    homog_points = triangulation.transpose()

                    get_points = cv2.convertPointsFromHomogeneous(homog_points)

                    print('get_points(solvePnP)\n', get_points)


                    # ToDo
                    # recoverPose + triangulatePoints
                    # E, mask = cv2.findEssentialMat(prime_C_u, list_2d_undistorted_o, cameraK, cv2.RANSAC, 0.999, 1,
                    #                                None)

                    E, mask = cv2.findEssentialMat(prime_C_d, blobs_2d_distort_o,
                                                   threshold=0.05,
                                                   prob=0.99,
                                                   method=cv2.RANSAC,
                                                   focal=focal,
                                                   pp=pp)

                    decompose = cv2.decomposeEssentialMat(E)
                    print('decompose\n', decompose)

                    p1, p2 = cv2.correctMatches(E, prime_C_u, list_2d_undistorted_o)
                    print('before ', prime_C_u)
                    print('after ', p1)
                    _, Rvecs, Tvecs, M = cv2.recoverPose(E, p1, p1)

                    RodP, _ = cv2.Rodrigues(prime_R)
                    print('RodP\n', RodP)
                    RodC, _ = cv2.Rodrigues(r_o)
                    print('RodC\n', RodC)
                    rvecPNP = np.linalg.inv(RodP) * RodC
                    tvecPNP = RodC * t_o - RodP * prime_T

                    print('rvecPNP\n', rvecPNP)
                    print('tvecPNP\n', tvecPNP)

                    print('Rvecs\n', Rvecs)
                    print('Tvecs\n', Tvecs)
                    P = np.hstack((Rvecs, Tvecs))
                    # curr_proj_matrix = cv2.hconcat([Rvecs, Tvecs])
                    # print('curr proj m\n', curr_proj_matrix)
                    pm1 = np.eye(3, 4)
                    # pm1 = np.hstack((np.eye(3), np.zeros((3, 1))))
                    triangulation_r = cv2.triangulatePoints(pm1, P,
                                                            prime_C_u,
                                                            list_2d_undistorted_o)
                    homog_points_r = triangulation_r.transpose()
                    get_points = cv2.convertPointsFromHomogeneous(homog_points_r)



                    print('get_points(recoverPose)\n', get_points)

                    print('prime cam info\n', prime_cam_info)
                    tmp_point = []
                    for recover_points in get_points:
                        print(recover_points)
                        p_o = vector3(recover_points[0][0],
                                      recover_points[0][1],
                                      recover_points[0][2])
                        new_cam_pose = {'cidx': prime_cam_info['camera']['idx'], 'position': vector3(0,0,0),
                                        'orient': prime_cam_info['camera']['orient']}
                        # tmp_p_o = transfer_point_inverse(p_o, prime_cam_info['camera'])
                        tmp_p_o = transfer_point_inverse(p_o, prime_cam_info['camera'])
                        tmp_point.append([tmp_p_o.x, tmp_p_o.y, tmp_p_o.z])
                        print(tmp_p_o)
                    print(tmp_point)
                    for index, leds in enumerate(info['leds']):
                        print('remake ', get_points[index])
                        leds_data[origin][leds]['remake_3d'].append(
                            {'idx': leds, 'cam_l': prime_cam_id, 'cam_r': int(info['camera']['idx']),
                             'coord': get_points[index],
                             'coord_tmp': tmp_point[index]})
                    # em, mask = cv2.findEssentialMat(prime_C_d, blobs_2d_distort_o,
                    #                                 threshold=0.05,
                    #                                 prob=0.95,
                    #                                 method=cv2.RANSAC,
                    #                                 focal=focal,
                    #                                 pp=pp)
                    # print('mask\n', mask)
                    # print('decompose\n', cv2.decomposeEssentialMat(em))
                    # F = np.linalg.inv(temp_camera_k.T).dot(em).dot(np.linalg.inv(temp_camera_k))
                    # # optimal solution for triangulation of  object points
                    # print('prime C U\n', prime_C_u)
                    # print('curr point\n', list_2d_undistorted_o)
                    # p1, p2 = cv2.correctMatches(em, prime_C_u, list_2d_undistorted_o)
                    #
                    # print('r_o\n', r_o)
                    # print('t_o\n', t_o)
                    #
                    # RodP, _ = cv2.Rodrigues(prime_R)
                    # print('RodP\n', RodP)
                    # RodC, _ = cv2.Rodrigues(r_o)
                    # print('RodC\n', RodC)
                    #
                    # rvecPNP = np.linalg.inv(RodP) * RodC
                    # tvecPNP = RodC * t_o - RodP * prime_T
                    #
                    # print('rvecPNP\n', rvecPNP)
                    # print('tvecPNP\n', tvecPNP)
                    #
                    # points, r, t, mask = cv2.recoverPose(em, p1, p2)
                    # print('r\n', r)
                    # print('t\n', t)
                    #
                    # P = np.hstack((r, t))
                    # pm1 = np.eye(3, 4)
                    #
                    # pts_sps = compute3Dpoints(pm1, P, p1, p2)
                    #
                    # print('result\n', pts_sps)


def compute3Dpoints(P1, P2, npts1, npts2):
    # computes object point coordinates from photo points correspondence using DLT method
    ptsh3d = cv2.triangulatePoints(P1, P2, npts1.T, npts2.T).T

    pts_sps = copy.deepcopy(ptsh3d[:, :3])
    for i, pt in enumerate(ptsh3d):
        pt = (pt / pt[3])[:3]
        pts_sps[i, :] = pt[:3]

    return pts_sps


def undistortPoints(pts, cam_m, distor):
    # returns normalized coordinates of photo points
    tp = np.expand_dims(pts, axis=0)
    tp_u = cv2.undistortPoints(tp, cam_m, distor)
    return tp_u


def UndistorTiePoints(tie_pts, cam_m, distor):
    # returns coordinates in photo system without effect of distortion
    tp1_u = undistortPoints(tie_pts[:, :2], cam_m, distor)
    tp2_u = undistortPoints(tie_pts[:, 2:4], cam_m, distor)

    shape = (1, len(tie_pts), 2)
    tp1 = np.empty(shape)
    tp2 = np.empty(shape)
    tp1[0, :, 0] = cam_m[0, 0] * tp1_u[0, :, 0] + cam_m[0, 2]
    tp1[0, :, 1] = cam_m[1, 1] * tp1_u[0, :, 1] + cam_m[1, 2]

    tp2[0, :, 0] = cam_m[0, 0] * tp2_u[0, :, 0] + cam_m[0, 2]
    tp2[0, :, 1] = cam_m[1, 1] * tp2_u[0, :, 1] + cam_m[1, 2]

    return tp1, tp2, tp1_u, tp2_u


def draw_blobs(ax, camera_array, origin, target):
    draw_dots(3, leds_data[origin], ax, 'blue')
    draw_dots(3, leds_data[target], ax, 'red')

    # # 원점
    # ax.scatter(0, 0, 0, marker='o', color='k', s=20)
    # ax.set_xlim([-0.8, 0.8])
    # ax.set_xlabel('X')
    # ax.set_ylim([-0.8, 0.8])
    # ax.set_ylabel('Y')
    # ax.set_zlim([-0.8, 0.8])
    # ax.set_zlabel('Z')
    # scale = 1.5
    # f = zoom_factory(ax, base_scale=scale)


def sliding_window(elements, window_size):
    key_array = []
    for i in range(len(elements) - window_size + 1):
        temp = elements[i:i + window_size]
        key_array.append(temp)

    return key_array


def draw_ax_plot(pts, ax, c):
    for i in range(len(pts)):
        if pts[i]['remake_3d'] != 'error':
            print(pts[i]['remake_3d'])
            x = [coord['coord'][0][0] for coord in pts[i]['remake_3d']]
            y = [coord['coord'][0][1] for coord in pts[i]['remake_3d']]
            z = [coord['coord'][0][2] for coord in pts[i]['remake_3d']]
            idx = [coord['idx'] for coord in pts[i]['remake_3d']]

            ax.scatter(x, y, z, marker='o', s=30, color=c, alpha=0.5)
            for idx, x, y, z in zip(idx, x, y, z):
                label = '%s' % idx
                ax.text(x, y, z, label, size=10)

            x = [coord['coord_tmp'][0] for coord in pts[i]['remake_3d']]
            y = [coord['coord_tmp'][1] for coord in pts[i]['remake_3d']]
            z = [coord['coord_tmp'][2] for coord in pts[i]['remake_3d']]
            idx = [coord['idx'] for coord in pts[i]['remake_3d']]

            ax.scatter(x, y, z, marker='o', s=30, color='blue', alpha=0.5)
            for idx, x, y, z in zip(idx, x, y, z):
                label = '%s' % idx
                ax.text(x, y, z, label, size=10)

    # 원점
    ax.scatter(0, 0, 0, marker='o', color='k', s=20)
    ax.set_xlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylim([-1, 1])
    ax.set_ylabel('Y')
    ax.set_zlim([-1, 1])
    ax.set_zlabel('Z')
    scale = 1.5
    f = zoom_factory(ax, base_scale=scale)


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

    # path_1 = "../cal_uvc/jsons/"
    # path_2 = '../cal_uvc/jsons/backup/good_6_2/'
    # path_3 = '../cal_uvc/jsons/backup/static_6/'
    # path_4 = '../cal_uvc/jsons/backup/static_6_2/'
    # path_5 = '../cal_uvc/jsons/backup/static_6_2_disable_refine/'
    camera_array_origin = init_model_json("../Calibration/jsons/")
    # camera_array_target = init_model_json(path_4)

    essential_test(camera_array_origin, origin, target)

    draw_ax_plot(leds_data[origin], ax, 'gray')
    draw_blobs(ax, camera_array_origin, origin, target)
    plt.show()
