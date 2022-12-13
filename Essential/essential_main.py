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

CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
angle_spec = 80


def make_camera_array():
    dist_to_controller = 0.5

    depth = 0.5
    step = 5
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


def get_facing_dot(fname, cam_pose):
    pts_facing = []
    leds_array = []

    for i, data in enumerate(leds_dic[fname]):
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


def calc_led_datas(info, temp_camera_k, temp_dist_coeff):
    # print(info)
    origin_leds = []

    for led_num in info['leds']:
        origin_leds.append([leds_dic[origin][led_num]['pos']])
    origin_leds = np.array(origin_leds, dtype=np.float64)
    # print(origin_leds[0][0][0])
    # 여기서 오리지널 pts를 투영 시켜 본다.
    obj_cam_pos_n = np.array(info['camera']['position'])
    obj_cam_pos_view = np.array(info['camera']['position_view'])
    rotR = R.from_quat(np.array(info['camera']['orient']))
    euler_degree = np.round_(get_euler_from_quat('zxy', info['camera']['orient']), 3)
    euler_radian = []
    for data in euler_degree:
        euler_radian.append(math.radians(data))

    print('euler_radian: ', euler_radian)
    # Rod, _ = cv2.Rodrigues(rotR.as_matrix())
    # print('Rod\n', Rod)

    oret = cv2.projectPoints(origin_leds, np.array(euler_radian), obj_cam_pos_view,
                             temp_camera_k,
                             None)
    o_xx, o_yy = oret[0].reshape(len(origin_leds), 2).transpose()
    blobs_2d_origin = []
    for i in range(len(o_xx)):
        blobs_2d_origin.append([[o_xx[i], o_yy[i]]])

    # print(blobs_2d_origin)
    blobs_2d_distort_o = np.array(blobs_2d_origin, dtype=np.float64)
    blobs_2d_distort_o = copy.deepcopy(np.reshape(blobs_2d_distort_o, (1, 6, 2)))

    # pts1 = np.ascontiguousarray(blobs_2d_distort_o, np.float32)

    list_2d_undistorted_o = cv2.undistortPoints(blobs_2d_distort_o,
                                                temp_camera_k,
                                                None)
    list_2d_undistorted_o = copy.deepcopy(np.reshape(list_2d_undistorted_o, (1, 6, 2)))

    _, r_o, t_o, inliers = cv2.solvePnPRansac(origin_leds, list_2d_undistorted_o,
                                              cameraK,
                                              distCoeff)

    return blobs_2d_distort_o, list_2d_undistorted_o, r_o, t_o, euler_radian, origin_leds


def get_cam_id(acc_cams_array, prime_num):
    for prime_cnt, info in enumerate(acc_cams_array):
        if int(info['camera']['idx']) == prime_num:
            return prime_cnt


cam_pose_map = {}
bridge_point = []
cam_pose_refactor = {}
cam_mesh_map = {}
primary_cam_array = []


def essential_test(cam_array, origin):
    cam_pose = make_camera_array()

    for i, data in enumerate(cam_pose):
        pts_facing, leds_array = get_facing_dot(origin, data)
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

    mesh_keys = sliding_window([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                0, 1, 2, 3, 4], 6)

    # find bridge Point
    for idx in range(len(mesh_keys) - 1):
        curr_key = ','.join(str(e) for e in mesh_keys[idx])
        next_key = ','.join(str(e) for e in mesh_keys[idx + 1])

        print('curr_key ', curr_key)
        print('next_key ', next_key)

        curr_cam_mesh_data = cam_pose_map.get(curr_key)
        next_cam_mesh_data = cam_pose_map.get(next_key)

        m_array = []
        for curr_cam_info in curr_cam_mesh_data:
            curr_cam_number = curr_cam_info['camera']['idx']
            for next_cam_info in next_cam_mesh_data:
                next_cam_number = next_cam_info['camera']['idx']
                if curr_cam_number == next_cam_number:
                    m_array.append({'camera': cam_pose[curr_cam_number]})
        bridge_point.append({'curr_key': curr_key, 'next_key': next_key, 'group': idx + 1, 'm_array': m_array})

    # ToDo
    ###################################################################

    # # init start position group
    # cam_mesh_data = bridge_point[0]['m_array']
    # min_position = find_short_path(cam_mesh_data, pos_cnt)
    # pos_cnt += 1
    # primary_cam_array.append({'prime_cam': min_position, 'acc_cam': cam_mesh_data, 'key': key_data})
    # draw_camera_position(cam_mesh_data[min_position], 'black')
    #

    start_cams_array = cam_pose_map.get("0,1,2,3,4,5")
    start_cam_num = get_cam_id(start_cams_array, 1319)
    start_cam_id, start_min_pos = find_short_path(start_cams_array, 0, start_cam_num, 0)

    print('start cam id', start_cam_id, ' array num ', start_min_pos)
    primary_cam_array.append({'prime_cam': start_cam_id, 'acc_cam': start_cams_array, 'key': [0,1,2,3,4,5]})
    draw_camera_position(start_cams_array[start_min_pos], 'black')

    pos_cnt = 1
    for key_data in mesh_keys:
        key_string = ','.join(str(e) for e in key_data)
        for curr_bridge in bridge_point:
            if key_string == curr_bridge['next_key']:
                cam_mesh_data = curr_bridge['m_array']

                # print('cam_mesh_data ', cam_mesh_data)

                # acc_data = cam_mesh_data[0]['acc_data']
                if len(cam_mesh_data) > 0:
                    min_pos_cam_id, min_position = find_short_path(cam_mesh_data, pos_cnt, NOT_SET, curr_bridge['group'])
                    print('key ', key_string)
                    print('group ', curr_bridge['group'])
                    pos_cnt += 1
                    primary_cam_array.append({'prime_cam': min_pos_cam_id, 'acc_cam': cam_mesh_data, 'key': key_data})
                    draw_camera_position(cam_mesh_data[min_position], 'black')


    print('primary len ', len(primary_cam_array))
    ###################################################################

    for i, cam_data in enumerate(primary_cam_array):
        key = cam_data['key']
        key_string = ','.join(str(e) for e in key)
        # ToDo
        # acc_cams_array = cam_data['acc_cam']
        acc_cams_array = cam_pose_map.get(key_string)

        primary_cam_number = get_cam_id(acc_cams_array, cam_data['prime_cam'])
        # primary_cam_number = cam_data['prime_cam']
        print('cam id ', cam_data['prime_cam'], ' tranlate id ', primary_cam_number)

        length = len(acc_cams_array)

        cam_num = 0
        temp_camera_k = cam_array[cam_num]['cam_cal']['cameraK']
        temp_dist_coeff = cam_array[cam_num]['cam_cal']['dist_coeff']
        pp = (temp_camera_k[0, 2], temp_camera_k[1, 2])
        focal = temp_camera_k[0, 0]
        # print('focal ', focal, ' pp ', pp)

        if length > 1:
            # if key != "0,1,11,12,13,14":
            #     continue
            # print('group ', i, ' len ', length)
            # print('leds ', key)

            # if key in cam_mesh_map:
            #     cam_mesh_map[key].append({'acc_data': acc_cams_array, 'group': i})
            # else:
            #     cam_mesh_map[key] = [{'acc_data': acc_cams_array, 'group': i}]

            # debug_number = get_cam_id(acc_cams_array, 2149)
            # debug_number = 0

            prime_C_d, prime_C_u, prime_R, prime_T, prime_cam_r, origin_leds = calc_led_datas(
                acc_cams_array[primary_cam_number], temp_camera_k, None)
            prime_cam_t = np.array(acc_cams_array[primary_cam_number]['camera']['position'])
            prime_cam_t_view = np.array(acc_cams_array[primary_cam_number]['camera']['position_view'])
            prime_rotR = R.from_quat(np.array(acc_cams_array[primary_cam_number]['camera']['orient']))
            prime_cam_id = int(acc_cams_array[primary_cam_number]['camera']['idx'])
            # draw_camera_position(acc_cams_array[primary_cam_number], 'black')
            over_spec_cnt = 0

            for prime_cnt, info in enumerate(acc_cams_array):
                if prime_cnt == primary_cam_number:
                    continue
                distort_2d, undistort_2d, r_o, t_o, euler_radian, _ = calc_led_datas(info, temp_camera_k, None)
                obj_cam_pos_view = np.array(info['camera']['position_view'])

                rvec1to2, tvec1to2 = camera_displacement(np.array(prime_cam_r), np.array(euler_radian),
                                                         prime_cam_t_view,
                                                         obj_cam_pos_view)
                # print('solvePnP')
                camera_displacement(prime_R, r_o, prime_T, t_o)

                # F, mask = cv2.findFundamentalMat(prime_C_d, blobs_2d_distort_o, method=cv2.FM_RANSAC)
                # print('F\n', F)
                E, mask = cv2.findEssentialMat(prime_C_u, undistort_2d, 1.0, (0, 0),
                                               cv2.RANSAC, 0.999, 1, None)

                # F2 = np.linalg.inv(temp_camera_k.T).dot(E).dot(np.linalg.inv(temp_camera_k))

                # E2 = temp_camera_k.T.dot(F2).dot(temp_camera_k)
                # print('F2\n', F2)
                # print('E\n', E)
                # print('E2\n', E2)

                _, Rvecs, Tvecs, M = cv2.recoverPose(E, prime_C_u, undistort_2d)
                recover_rvec, _ = cv2.Rodrigues(Rvecs)
                # print('After Recover Pose\n')
                # print('Rvecs\n', recover_rvec.T)
                # print('Tvecs\n', Tvecs)
                scalePose = tvec1to2[2] / Tvecs[2][0]
                # print('scalePose : ', scalePose)

                new_Tvecs = scalePose * Tvecs
                # print('new_Tvecs\n', new_Tvecs.T)
                # trianglutatePoints
                # print('solvePnP + trianglutePoint')

                # ToDo
                # SolvePnP + triangluatePoints
                left_rotation, jacobian = cv2.Rodrigues(prime_R)
                right_rotation, jacobian = cv2.Rodrigues(r_o)

                left_projection = np.hstack((left_rotation, prime_T))
                # print('left_project\n', left_projection)
                right_projection = np.hstack((right_rotation, t_o))
                # print('right_project\n', right_projection)

                triangulation = cv2.triangulatePoints(left_projection, right_projection,
                                                      prime_C_u,
                                                      undistort_2d)
                homog_points = triangulation.transpose()

                sget_points = cv2.convertPointsFromHomogeneous(homog_points)
                sdistance = 0
                for ii, points in enumerate(sget_points):
                    sdistance += math.sqrt(math.pow(points[0][0] - origin_leds[ii][0][0], 2) +
                                           math.pow(points[0][1] - origin_leds[ii][0][1], 2) +
                                           math.pow(points[0][2] - origin_leds[ii][0][2], 2))

                solvepnp_distance = sdistance / len(sget_points)
                # print('solvepnp_distance ', solvepnp_distance)

                # print('get_points(solvePnP)\n', sget_points)

                # print('recoverPose + triangluatePoint')
                # P = np.hstack((Rvecs, new_Tvecs))
                #
                # pm1 = np.hstack((np.eye(3), np.zeros((3, 1))))
                # print('pm1\n', pm1)
                # print('P\n', P)
                d_r, _ = cv2.Rodrigues(np.eye(3))
                d_t = np.zeros((3, 1))
                #
                # print(camera_displacement(d_r, recover_rvec, d_t, new_Tvecs))

                rvec_1to2, tvec_1to2 = camera_displacement(d_r, recover_rvec, d_t, new_Tvecs)
                inv_rvec, inv_tvec = inverse_matrix(rvec_1to2, tvec_1to2.T[0], prime_R.T[0], prime_T.T[0])
                # print('r_2\n', r_o)
                # print('t_2\n', t_o)
                # print('inv_rvec\n', inv_rvec)
                # print('inv_tvec\n', inv_tvec)
                inv_right_rotation, jacobian = cv2.Rodrigues(inv_rvec)
                int_right_projection = np.hstack((inv_right_rotation, inv_tvec))
                # print('int_right_projection\n', int_right_projection)

                triangulation_r = cv2.triangulatePoints(left_projection, int_right_projection,
                                                        prime_C_u,
                                                        undistort_2d)
                homog_points_r = triangulation_r.transpose()
                get_points = cv2.convertPointsFromHomogeneous(homog_points_r)

                # print('get_points(recoverPose)\n', get_points)

                over_distance = 0
                distance = 0
                for ii, points in enumerate(get_points):
                    distance += math.sqrt(math.pow(points[0][0] - origin_leds[ii][0][0], 2) +
                                          math.pow(points[0][1] - origin_leds[ii][0][1], 2) +
                                          math.pow(points[0][2] - origin_leds[ii][0][2], 2))

                recover_distance = distance / len(get_points)
                # print('recover_distance ', recover_distance)

                # spec 2cm
                if recover_distance > 0.02:
                    over_distance = 1
                    over_spec_cnt += 1
                    # draw_camera_position(info, 'gray')

                if over_distance == 0:
                    draw_camera_position(info, 'red')
                    # ToDo
                    for index, leds in enumerate(info['leds']):
                        # print('remake ', get_points[index])
                        leds_dic[origin][leds]['remake_3d'].append(
                            {'idx': leds, 'cam_l': prime_cam_id, 'cam_r': int(info['camera']['idx']),
                             'solve_coord': sget_points[index],
                             'coord': get_points[index]})

                    if key_string in cam_pose_refactor:
                        cam_pose_refactor[key_string].append({'camera': info['camera'], 'leds': key})
                    else:
                        cam_pose_refactor[key_string] = [{'camera': info['camera'], 'leds': key}]


find_path_array = []
find_cam_id = []
def find_short_path(acc_data, idx, start_num, group):
    print('acc len ', len(acc_data))
    if len(acc_data) == 0:
        return NOT_SET, NOT_SET
    MIN_DIS = 1.0
    MAX_DIS = 0.0
    POSITION = 0
    prev_group = -1

    if idx != 0:
        prev_candidates_pts = find_path_array[idx - 1][0]['camera']['position_view']
        prev_group = find_path_array[idx - 1][1]
        for i, candidates in enumerate(acc_data):
            # draw_camera_position(candidates, 'red')
            candidates_pts = candidates['camera']['position_view']
            curr_idx = int(candidates['camera']['idx'])

            dis = math.sqrt(math.pow(prev_candidates_pts.x - candidates_pts.x, 2) +
                            math.pow(prev_candidates_pts.y - candidates_pts.y, 2) +
                            math.pow(prev_candidates_pts.z - candidates_pts.z, 2))
            if dis < MIN_DIS and not curr_idx in find_cam_id:
                MIN_DIS = copy.deepcopy(dis)
                POSITION = copy.deepcopy(i)
            # if dis > MAX_DIS and not curr_idx in find_cam_id:
            #     MAX_DIS = copy.deepcopy(dis)
            #     POSITION = copy.deepcopy(i)
    else:
        POSITION = start_num
        prev_candidates_pts = vector3(0, 0, 0)

    find_path_array.append([acc_data[POSITION], group])
    find_cam_id.append(int(acc_data[POSITION]['camera']['idx']))
    curr_candidates_pts = acc_data[POSITION]['camera']['position_view']

    x = np.array([prev_candidates_pts.x, curr_candidates_pts.x])
    y = np.array([prev_candidates_pts.y, curr_candidates_pts.y])
    z = np.array([prev_candidates_pts.z, curr_candidates_pts.z])
    # plotting
    if group - 1 == prev_group:
        ax.plot3D(x, y, z, linewidth=1)

    # return cam_id
    return int(acc_data[POSITION]['camera']['idx']), POSITION


def draw_camera_position(cam_data, color):
    new_pt = cam_data['camera']['position_view']
    ax.scatter(new_pt.x, new_pt.y, new_pt.z, marker='.', color=color, s=20)
    label = (f"{cam_data['camera']['idx']}")
    ax.text(new_pt.x, new_pt.y, new_pt.z, label, size=5)


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
    draw_dots(3, leds_dic[origin], ax, 'blue')
    # draw_dots(3, leds_data[target], ax, 'red')

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
    key_array_sorted = []
    for i in range(len(elements) - window_size + 1):
        temp = elements[i:i + window_size]
        key_array.append(temp)

    for keys in key_array:
        temp = copy.deepcopy(keys)
        temp.sort()
        key_array_sorted.append(temp)

    return key_array_sorted


def draw_ax_plot(pts, ax, c):
    for i in range(len(pts)):
        if pts[i]['remake_3d'] != 'error':
            # print(pts[i]['remake_3d'])
            x = [coord['coord'][0][0] for coord in pts[i]['remake_3d']]
            y = [coord['coord'][0][1] for coord in pts[i]['remake_3d']]
            z = [coord['coord'][0][2] for coord in pts[i]['remake_3d']]
            idx = [coord['idx'] for coord in pts[i]['remake_3d']]

            ax.scatter(x, y, z, marker='o', s=7, color=c, alpha=0.5)
            for idx, x, y, z in zip(idx, x, y, z):
                label = '%s' % idx
                ax.text(x, y, z, label, size=3)

            # x = [coord['coord_tmp'][0] for coord in pts[i]['remake_3d']]
            # y = [coord['coord_tmp'][1] for coord in pts[i]['remake_3d']]
            # z = [coord['coord_tmp'][2] for coord in pts[i]['remake_3d']]
            # idx = [coord['idx'] for coord in pts[i]['remake_3d']]
            #
            # ax.scatter(x, y, z, marker='o', s=10, color='blue', alpha=0.5)
            # for idx, x, y, z in zip(idx, x, y, z):
            #     label = '%s' % idx
            #     ax.text(x, y, z, label, size=5)

            # x = [coord['solve_coord'][0][0] for coord in pts[i]['remake_3d']]
            # y = [coord['solve_coord'][0][1] for coord in pts[i]['remake_3d']]
            # z = [coord['solve_coord'][0][2] for coord in pts[i]['remake_3d']]
            # idx = [coord['idx'] for coord in pts[i]['remake_3d']]
            #
            # ax.scatter(x, y, z, marker='o', s=10, color='black', alpha=0.5)
            # for idx, x, y, z in zip(idx, x, y, z):
            #     label = '%s' % idx
            #     ax.text(x, y, z, label, size=5)

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

    camera_array_origin = init_model_json("../stereo_calibration/", cam_json)

    essential_test(camera_array_origin, origin)

    draw_ax_plot(leds_dic[origin], ax, 'black')
    draw_blobs(ax, camera_array_origin, origin, target)
    plt.show()
