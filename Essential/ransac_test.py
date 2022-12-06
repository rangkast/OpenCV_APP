import copy
import itertools
import json
import math
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import re
from cal_def import *

leds_data = {'name': 'led_dictionary'}
pos_offset = {'position': vector3(0.0, 0.0, 0.0), 'orient': get_quat_from_euler('xyz', [10, 10, 15])}

cameraK = np.eye(3).astype(np.float64)
distCoeff = np.zeros((4, 1)).astype(np.float64)

uvc_cameraK = np.array([[712.623, 0.0, 653.448],
                        [0.0, 712.623, 475.572],
                        [0.0, 0.0, 1.0]], dtype=np.float64)
uvc_distCoeff = np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)
LEDS_COUNT = 15

cam_json = ['cam_110.json', 'cam_101.json',
            'cam_002.json', 'cam_010.json',
            'cam_011.json', 'cam_003.json',
            'cam_000.json', 'cam_001.json',
            'cam_111.json', 'cam_100.json'
            ]


def init_model_json(path):
    print('start ', init_model_json.__name__)
    camera_info_array = []
    global group_num
    try:
        A = []
        K = []
        RVECS = []
        TVECS = []

        for i in range(len(cam_json)):
            print('\n')
            print('cam_id[', i, ']')

            # ../ backup / jsons /
            jsonObject = json.load(
                open(''.join([path, f'{cam_json[i]}'])))
            model_points = jsonObject.get('model_points')
            model = []
            leds_array = []
            for data in model_points:
                idx = data.split('LED')[1]
                # ToDo
                x = model_points.get(data)[0]
                y = model_points.get(data)[1]
                s = model_points.get(data)[2]
                model.append({'idx': idx, 'x': x, 'y': y, 'size': s})
                leds_array.append(int(idx))

            f = jsonObject.get('camera_f')
            c = jsonObject.get('camera_c')
            k = jsonObject.get('camera_k')

            A = np.array([[f[0], 0.0, c[0]],
                          [0.0, f[1], c[1]],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
            K = np.array([[k[0]], [k[1]], [k[2]], [k[3]]], dtype=np.float64)

            rvecs = jsonObject.get('rvecs')
            tvecs = jsonObject.get('tvecs')
            RVECS = np.array([[rvecs[0]], [rvecs[1]], [rvecs[2]]], dtype=np.float64)
            TVECS = np.array([[tvecs[0]], [tvecs[1]], [tvecs[2]]], dtype=np.float64)

            blobs = jsonObject.get('blobs')
            undistorted_2d = jsonObject.get('undistort_blobs')
            distorted_2d = jsonObject.get('distorted_2d')
            # print('cameraK: ', A)
            # print('dist_coeff: ', K)
            #
            # print('rvecs: ', RVECS)
            # print('tvecs: ', TVECS)
            #
            # print('blobs: ', blobs)
            # print('undistorted: ', undistorted_2d)
            # print('distorted_2d: ', distorted_2d)

            cam_pos = TVECS.reshape(3)
            cam_ori = R.from_rotvec(RVECS.reshape(3)).as_quat()
            cam_ori_euler = np.round_(get_euler_from_quat('zxy', cam_ori), 3)
            cam_ori_quat = np.round_(get_quat_from_euler('zxy', cam_ori_euler), 8)

            # print('pos: ', cam_pos, ' ori(euler): ', cam_ori_euler,
            #       ' ori(quat): ', cam_ori,
            #       ' ori(re-quat): ', cam_ori_quat)

            if i % 2 == 0:
                group_num = i

            camera_info_array.append(
                {'cidx': i,
                 'position': vector3(cam_pos[0], cam_pos[1], cam_pos[2]),
                 'orient': quat(cam_ori[0], cam_ori[1], cam_ori[2], cam_ori[3]),
                 'euler': cam_ori_euler,
                 'pts_facing': {},
                 'coord': {'undistort_r': [], 'distort_r': [], 'undistort_s': [], 'distort_s': [],
                           'remake_undistort_r': [], 'remake_undistort_s': []},
                 'rt': {'rvecs_r': [], 'tvecs_r': [], 'rvecs_s': [], 'tvecs_s': [],
                        'r_rvecs_r': [], 'r_tvecs_r': [], 'r_rvecs_s': [], 'r_tvecs_s': []},
                 'test': [],
                 'json': cam_json[i],
                 'group': group_num,
                 'blobs': blobs,
                 'distorted_2d': distorted_2d,
                 'undistorted_2d': undistorted_2d,
                 'cam_cal': {'cameraK': A, 'dist_coeff': K},
                 'model': {'leds': leds_array, 'spec': model},
                 'S_R_T': {'rvecs': RVECS, 'tvecs': TVECS},
                 'rer_info': {'min': 10.0, 'pos': None}
                 }
            )
    except:
        print('exception')
        traceback.print_exc()
    else:
        print('done')
    finally:
        return camera_info_array


def read_led_pts(fname):
    pts = []
    with open(f'{fname}.txt', 'r') as F:
        a = F.readlines()
    for idx, x in enumerate(a):
        m = re.match(
            '\{ \.pos *= \{+ *(-*\d+.\d+),(-*\d+.\d+),(-*\d+.\d+) *\}+, \.dir *=\{+ *(-*\d+.\d+),(-*\d+.\d+),(-*\d+.\d+) *\}+, \.pattern=(\d+) },',
            x)
        x = float(m.group(1))
        y = float(m.group(2))
        z = float(m.group(3))
        u = float(m.group(4))
        v = float(m.group(5))
        w = float(m.group(6))
        _pos = list(map(float, [x, y, z]))
        _dir = list(map(float, [u, v, w]))
        pts.append({'idx': idx, 'pos': _pos, 'dir': _dir, 'pair_xy': [], 'remake_3d': [],
                    'min': {'dis': 10, 'remake': []}})

    # print(f'{fname} PointsRead')
    leds_data[fname] = pts


def draw_blobs(ax, camera_array, origin, target):
    draw_dots(3, leds_data[origin], ax, 'blue')
    draw_dots(3, leds_data[target], ax, 'red')

    # 원점
    ax.scatter(0, 0, 0, marker='o', color='k', s=20)
    ax.set_xlim([-0.7, 0.7])
    ax.set_xlabel('X')
    ax.set_ylim([-0.7, 0.7])
    ax.set_ylabel('Y')
    ax.set_zlim([-0.7, 0.7])
    ax.set_zlabel('Z')
    scale = 1.5
    f = zoom_factory(ax, base_scale=scale)

    # for idx, cam_pose in enumerate(camera_array):
    #     new_cam_pose = {'cidx': cam_pose['cidx'], 'position': cam_pose['position'],
    #                     'orient': get_quat_from_euler('xyz', [90,
    #                                                           0,
    #                                                           get_euler_from_quat('zxy', cam_pose['orient'])[2]])}
    #     new_pt = camtoworld(new_cam_pose['position'], new_cam_pose)
    #     ax.scatter(new_pt.x, new_pt.y, new_pt.z, marker='x', color='blue', s=20)
        # idx_array = []
        # for facing_data in cam_pose['pts_facing']['origin']:
        #     idx_array.append(int(facing_data['idx']))
        # nidx_array = []
        # for facing_data in cam_pose['pts_facing']['real']:
        #     nidx_array.append(int(facing_data['idx']))
        # str = ''.join([f"{new_cam_pose['cidx']}\n", f"{idx_array}", '\n', f"{nidx_array}"])
        # ax.text(new_pt.x, new_pt.y, new_pt.z, str, size=7)


#
def make_camera_array():
    MAX_DEGREE = 6
    cam_id = 0
    for idx in range(MAX_DEGREE):
        degree = idx * 30

        camera_array.append(
            {'cidx': cam_id,
             'position': vector3(0.0, -0.0, 0.3),
             'orient': get_quat_from_euler('zxy', [0, 55, degree]),
             'pts_facing': {},
             'distort': [],
             'undistort': [],
             'rvecs': [],
             'tvecs': [],
             'urvecs': [],
             'utvecs': [],
             'test': []})
        camera_array.append(
            {'cidx': cam_id + 1,
             'position': vector3(0.0, -0.0, 0.3),
             'orient': get_quat_from_euler('zxy', [0, 55, -(180 - degree)]),
             'pts_facing': {},
             'distort': [],
             'undistort': [],
             'rvecs': [],
             'tvecs': [],
             'urvecs': [],
             'utvecs': [],
             'test': []})
        cam_id += 2


#
# def make_camera_array():
#     cam_id = 0
#     camera_array.append(
#         {'cidx': cam_id, 'position': vector3(0.0, 0.0, 0.7),
#          'orient': get_quat_from_euler('zxy', [0, 55, -110]),
#          'pts_facing': {},
#          'distort': [],
#          'undistort': [],
#          'rvecs': [],
#          'tvecs': [],
#          'urvecs': [],
#          'utvecs': [],
#          'test': []})
#     camera_array.append(
#         {'cidx': cam_id + 1, 'position': vector3(0.0, 0.0, 0.7),
#          'orient': get_quat_from_euler('zxy', [0, 55, -150]),
#          'pts_facing': {},
#          'distort': [],
#          'undistort': [],
#          'rvecs': [],
#          'tvecs': [],
#          'urvecs': [],
#          'utvecs': [],
#          'test': []})


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


def cal_RER_px(led_ids, points3D, points2D, inliers, rvecs, tvecs, camera_k, dist_coeff):
    # print(points2D)
    temp_points2D = []
    # for u_data in points2D:
    #     temp_points2D.append([u_data[0][0], u_data[0][1]])
    # points2D = np.array(temp_points2D)

    points2D_reproj = cv2.fisheye.projectPoints(points3D, rvecs,
                                                tvecs, camera_k, dist_coeff)[0].squeeze(1)
    # print('points2D_reproj\n', points2D_reproj, '\npoints2D\n', points2D, '\n inliers: ', inliers)
    assert (points2D_reproj.shape == points2D.shape)
    error = (points2D_reproj - points2D)[inliers]  # Compute error only over inliers.
    rmse = 0.0

    for i in range(len(error[:, 0])):
        diff = np.power(error[:, 0][i][0], 2) + np.power(error[:, 0][i][1], 2)
        # print('led: ', led_ids[i], ' diff: ', '%0.12f' % diff)
        rmse += diff
    if inliers is None:
        return -1
    RER = round(np.sqrt(rmse) / (len(inliers)), 12)

    return RER


def find_good_pos(ax, cam_array_data, cam_pose_data, color, toggle, draw, RER, rad_sum):
    new_cam_pose = {'cidx': cam_pose_data['cidx'], 'position': cam_pose_data['position'],
                    'orient': get_quat_from_euler('xyz', [90, 0,
                                                          get_euler_from_quat('zxy', cam_pose_data['orient'])[2]])}
    new_pt = camtoworld(new_cam_pose['position'], new_cam_pose)

    if RER < cam_array_data['rer_info']['min']:
        cam_array_data['rer_info']['min'] = RER
        cam_array_data['rer_info']['pos'] = cam_pose_data

    if draw == 1:
        if toggle == 1:
            ax.scatter(new_pt.x + 0.01, new_pt.y, new_pt.z, marker='o', color=color, s=5)
            ax.text(new_pt.x + 0.01, new_pt.y, new_pt.z, cam_pose_data['cidx'], size=7)
        else:
            ax.scatter(new_pt.x, new_pt.y, new_pt.z, marker='o', color=color, s=5)
            ax.text(new_pt.x, new_pt.y, new_pt.z, cam_pose_data['cidx'], size=7)


def facing_dot_area(origin):
    cam_id_array = []
    rer_array = []
    drer_array = []
    MAX_DEGREE = 36
    cam_id = 0
    cam_pose = []
    # 30cm
    for dist in range(1):
        for idx in range(MAX_DEGREE):
            degree = idx * 5
            cam_pose.append({
                'cidx': cam_id,
                'position': vector3(0.0, 0.0, 0.3 + (dist * 0.1)),
                'orient': get_quat_from_euler('zxy', [0, 55, degree])
            })
            cam_pose.append({
                'cidx': cam_id + 1,
                'position': vector3(0.0, 0.0, 0.3 + (dist * 0.1)),
                'orient': get_quat_from_euler('zxy', [0, 55, -(180 - degree)])
            })
            cam_id += 2
    for cam_array_data in camera_array:
        model = np.array(cam_array_data['model']['leds'])
        group_num = cam_array_data['group']
        temp_camera_k = cam_array_data['cam_cal']['cameraK']
        temp_dist_coeff = cam_array_data['cam_cal']['dist_coeff']
        TVECS = cam_array_data['S_R_T']['tvecs'].reshape(3)
        # print('m: ', model)
        for i, cam_pose_data in enumerate(cam_pose):
            o_pts_facing, leds_array = np.array(check_facing_dot(leds_data[origin], cam_pose_data))
            cnt = 0

            new_pts_facing = []

            for i, led_num in enumerate(leds_array):
                if led_num in model:
                    new_pts_facing.append(o_pts_facing[i])
                    cnt += 1

            if cnt == len(model):
                # print('f: ', leds_array)
                # print('cam id: ', cam_pose_data['cidx'], ' ori: ', cam_pose_data)
                # print('n ', new_pts_facing)
                # facing dot 과 좌표를 찾는다.
                new_pts_facing = np.array(new_pts_facing)
                origin_leds = []
                angle = []
                for x in new_pts_facing:
                    origin_leds.append([[x['pos'][0], x['pos'][1], x['pos'][2]]])
                    angle.append(x['angle'])
                origin_leds = np.array(origin_leds, dtype=np.float64)

                # 여기서 오리지널 pts를 투영 시켜 본다.
                # cam_pose_data['position'].x = round(copy.deepcopy(TVECS[0]), 8)
                cam_pose_data['position'].y = round(copy.deepcopy(TVECS[1]), 8)
                cam_pose_data['position'].z = round(copy.deepcopy(TVECS[2]), 8)
                obj_cam_pos_n = np.array(cam_pose_data['position'])
                rotR = R.from_quat(np.array(cam_pose_data['orient']))
                Rod, _ = cv2.Rodrigues(rotR.as_matrix())
                oret = cv2.projectPoints(origin_leds, Rod, obj_cam_pos_n,
                                         temp_camera_k,
                                         temp_dist_coeff)
                o_xx, o_yy = oret[0].reshape(len(origin_leds), 2).transpose()
                blobs_2d_origin = []
                for i in range(len(o_xx)):
                    blobs_2d_origin.append([[o_xx[i], o_yy[i]]])

                blobs_2d = np.array(blobs_2d_origin, dtype=np.float64)
                list_2d_undistorted = cv2.fisheye.undistortPoints(blobs_2d,
                                                                  temp_camera_k,
                                                                  temp_dist_coeff)

                blobs_2d = blobs_2d[:, 0]
                list_2d_undistorted = list_2d_undistorted[:, 0]

                # solvePnPRansac으로 test
                check = len(origin_leds)
                if check != len(blobs_2d):
                    print("assertion not equal: ", len(blobs_2d))
                    continue

                if check < 4 or len(blobs_2d) < 4:
                    print("assertion < 4: ", check)
                    continue

                _, rvecs, tvecs, inliers = cv2.solvePnPRansac(origin_leds, blobs_2d,
                                                              temp_camera_k,
                                                              temp_dist_coeff, useExtrinsicGuess=True,
                                                              iterationsCount=100,
                                                              confidence=0.99,
                                                              reprojectionError=8.0,
                                                              flags=cv2.SOLVEPNP_ITERATIVE)
                RER = '%0.12f' % cal_RER_px(model,
                                            origin_leds,
                                            blobs_2d,
                                            inliers,
                                            rvecs,
                                            tvecs,
                                            temp_camera_k,
                                            temp_dist_coeff)
                # default
                _, drvecs, dtvecs, dinliers = cv2.solvePnPRansac(origin_leds, list_2d_undistorted,
                                                                 cameraK,
                                                                 distCoeff, useExtrinsicGuess=True,
                                                                 iterationsCount=100,
                                                                 confidence=0.99,
                                                                 reprojectionError=8.0,
                                                                 flags=cv2.SOLVEPNP_ITERATIVE)
                dRER = '%0.12f' % cal_RER_px(model,
                                             origin_leds,
                                             list_2d_undistorted,
                                             dinliers,
                                             drvecs,
                                             dtvecs,
                                             cameraK,
                                             distCoeff)
                # print('RER: ', RER, ' dRER: ', dRER)

                if float(RER) != -1 and float(dRER) != -1 and float(RER) < 0.06:
                    cam_id_array.append(group_num)
                    rer_array.append(float(RER))
                    rad_sum = 0
                    for index in range(len(model) - 1):
                        # led: 0 - 0.998154062220473
                        # led: 1 - 0.9326896028775354
                        # led: 2 - 0.6761841884745897
                        # led: 3 - 0.49133407944570695 제외
                        print('led : ', model[index], angle[index])
                        rad_sum += angle[index]
                    print(cam_pose_data['cidx'], ':', np.round_(get_euler_from_quat('zxy', cam_pose_data['orient']), 3),
                          ' ',
                          np.round_(cam_pose_data['position'], 3))
                    print('RER: ', RER, ' dRER: ', dRER)
                    # drer_array.append(dRER)

                    if group_num == 0:
                        if cam_array_data['cidx'] % 2 == 0:
                            find_good_pos(ax, cam_array_data, cam_pose_data, 'red', 1, 1, float(RER), rad_sum)
                        else:
                            find_good_pos(ax, cam_array_data, cam_pose_data, 'red', 1, 0, float(RER), rad_sum)
                    if group_num == 2:
                        if cam_array_data['cidx'] % 2 == 0:
                            find_good_pos(ax, cam_array_data, cam_pose_data, 'blue', 0, 1, float(RER), rad_sum)
                        else:
                            find_good_pos(ax, cam_array_data, cam_pose_data, 'blue', 0, 0, float(RER), rad_sum)
                    if group_num == 4:
                        if cam_array_data['cidx'] % 2 == 0:
                            find_good_pos(ax, cam_array_data, cam_pose_data, 'magenta', 1, 1, float(RER), rad_sum)
                        else:
                            find_good_pos(ax, cam_array_data, cam_pose_data, 'magenta', 1, 0, float(RER), rad_sum)
                    if group_num == 6:
                        if cam_array_data['cidx'] % 2 == 0:
                            find_good_pos(ax, cam_array_data, cam_pose_data, 'green', 0, 1, float(RER), rad_sum)
                        else:
                            find_good_pos(ax, cam_array_data, cam_pose_data, 'green', 0, 0, float(RER), rad_sum)
                    if group_num == 8:
                        if cam_array_data['cidx'] % 2 == 0:
                            find_good_pos(ax, cam_array_data, cam_pose_data, 'purple', 1, 1, float(RER), rad_sum)
                        else:
                            find_good_pos(ax, cam_array_data, cam_pose_data, 'purple', 1, 0, float(RER), rad_sum)

        print('cam_id ', cam_array_data['cidx'])
        # print(cam_array_data['rer_info']['pos']['cidx'])
        # print('%0.12f' % cam_array_data['rer_info']['min'])
        # print(cam_array_data['rer_info']['pos']['position'])
        # print(np.round_(get_euler_from_quat('zxy', cam_array_data['rer_info']['pos']['orient']), 8))
        print('\n')

        # new_cam_pose = {'cidx': cam_pose_data['cidx'], 'position': cam_array_data['rer_info']['pos']['position'],
        #                 'orient': get_quat_from_euler('xyz',
        #                                               [90,
        #                                                0,
        #                                                get_euler_from_quat('zxy',
        #                                                                    cam_array_data['rer_info']['pos']['orient'])[
        #                                                    2]])}
        # new_pt = camtoworld(new_cam_pose['position'], new_cam_pose)
        # ax.scatter(new_pt.x, new_pt.y, new_pt.z, marker='*', color='blue', s=20)
        # ax.text(new_pt.x, new_pt.y, new_pt.z, cam_array_data['cidx'], size=20)

    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.style.use('default')
    plt.figure(figsize=(15, 10))
    plt.title('RER')
    markers, stemlines, baseline = plt.stem(cam_id_array, rer_array)
    markers.set_color('red')


def ransac_test(origin, target):
    plt.style.use('default')
    plt.rc('xtick', labelsize=5)  # x축 눈금 폰트 크기
    plt.rc('ytick', labelsize=5)
    fig_ransac = plt.figure(figsize=(15, 15))
    rer_array = []
    cam_id = []
    cnt = 0
    normal_rer = []
    default_rer = []
    for idx, cam_pose in enumerate(camera_array):
        print('##############################')
        print('cam id: ', idx)
        fig_ransac.tight_layout()
        ax_ransac = fig_ransac.add_subplot(4, round((len(camera_array) + 1) / 4), idx + 1)
        o_pts_facing, _ = check_facing_dot(leds_data[origin], cam_pose)
        o_pts_facing = np.array(o_pts_facing)
        # origin_leds = np.array([[x['pos'][0], x['pos'][1], x['pos'][2]] for x in o_pts_facing])

        # 여기서 facing dot을 한번 섞어봄
        new_pts_facing = []
        for i in range(len(cam_pose['model']['leds'])):
            for pts in o_pts_facing:
                if int(pts['idx']) == int(cam_pose['model']['leds'][i]):
                    new_pts_facing.append(pts)
                    break
        # print('n ', new_pts_facing)
        # print('o ', o_pts_facing)
        new_pts_facing = np.array(new_pts_facing)
        origin_leds = []
        for x in new_pts_facing:
            origin_leds.append([[x['pos'][0], x['pos'][1], x['pos'][2]]])
        origin_leds = np.array(origin_leds, dtype=np.float64)

        t_pts_facing, _ = check_facing_dot(leds_data[target], cam_pose)
        t_pts_facing = np.array(t_pts_facing)
        noise_leds = []
        for x in t_pts_facing:
            noise_leds.append([[x['pos'][0], x['pos'][1], x['pos'][2]]])
        noise_leds = np.array(noise_leds, dtype=np.float64)
        cam_pose['pts_facing'] = {'origin': o_pts_facing, 'target': t_pts_facing, 'real': new_pts_facing}

        # check led id array
        led_arrays = []
        for led_num in new_pts_facing:
            led_arrays.append(led_num['idx'])

        obj_cam_pos_n = np.array(cam_pose['position'])
        rotR = R.from_quat(np.array(cam_pose['orient']))
        Rod, _ = cv2.Rodrigues(rotR.as_matrix())

        oret = cv2.fisheye.projectPoints(origin_leds,
                                         cam_pose['S_R_T']['rvecs'],
                                         cam_pose['S_R_T']['tvecs'],
                                         cam_pose['cam_cal']['cameraK'],
                                         cam_pose['cam_cal']['dist_coeff'])

        o_xx, o_yy = oret[0].reshape(len(origin_leds), 2).transpose()

        blobs_2d_origin = []
        for i in range(len(o_xx)):
            blobs_2d_origin.append([[o_xx[i], o_yy[i]]])

        blobs_2d = np.array(blobs_2d_origin, dtype=np.float64)
        list_2d_undistorted = cv2.fisheye.undistortPoints(blobs_2d,
                                                          cam_pose['cam_cal']['cameraK'],
                                                          cam_pose['cam_cal']['dist_coeff'])

        # 실측 한거
        real_blobs_2d = np.array(cam_pose['distorted_2d'], dtype=np.float64)
        real_list_2d_undistorted = np.array(cam_pose['undistorted_2d'], dtype=np.float64)
        # 시뮬레이션
        blobs_2d = blobs_2d[:, 0]
        list_2d_undistorted = list_2d_undistorted[:, 0]

        print('u ', real_list_2d_undistorted)
        print('u ', list_2d_undistorted)

        # for index, data in enumerate(real_list_2d_undistorted):
        #     # diff_x = '%0.12f' % (data[0] - list_2d_undistorted[index][0])
        #     # diff_y = '%0.12f' % (data[1] - list_2d_undistorted[index][1])
        #     # print(led_arrays[index], ':', diff_x, ',', diff_y)
        #
        #     if idx == 0:
        #         if led_arrays[index] == 0:
        #             diff_x , diff_y = -0.000000087212, -0.000000540346
        #         if led_arrays[index] == 1:
        #             diff_x , diff_y = 0.000000395694, 0.000001578344
        #         if led_arrays[index] == 2:
        #             diff_x , diff_y = -0.000000299019, -0.000000986780
        #         if led_arrays[index] == 3:
        #             diff_x , diff_y = 0.000727952600, 0.001836464688
        #
        #     if idx == 1:
        #         if led_arrays[index] == 0:
        #             diff_x , diff_y = 0.000001369094, 0.000001080298
        #         if led_arrays[index] == 1:
        #             diff_x , diff_y = 0.000000755972, -0.000000465921
        #         if led_arrays[index] == 2:
        #             diff_x , diff_y = -0.000002177463, -0.000000662857
        #         if led_arrays[index] == 3:
        #             diff_x , diff_y = 0.000064437528, 0.000217194874
        #
        #     if idx == 2:
        #         if led_arrays[index] == 9:
        #             diff_x , diff_y = -0.000010441910, 0.000003181484
        #         if led_arrays[index] == 10:
        #             diff_x , diff_y = 0.000002988502, -0.000010934687
        #         if led_arrays[index] == 11:
        #             diff_x , diff_y = 0.000007534694, 0.000007413093
        #         if led_arrays[index] == 12:
        #             diff_x , diff_y = -0.001344809168, 0.002776299455
        #
        #     if idx == 3:
        #         if led_arrays[index] == 9:
        #             diff_x , diff_y = 0.000000055888, 0.000000007284
        #         if led_arrays[index] == 10:
        #             diff_x , diff_y = 0.000000248929, -0.000000242167
        #         if led_arrays[index] == 11:
        #             diff_x , diff_y = -0.000000284699, 0.000000219058
        #         if led_arrays[index] == 12:
        #             diff_x , diff_y = -0.001043335990, -0.002212067974
        #
        #     if idx == 4:
        #         if led_arrays[index] == 14:
        #             diff_x , diff_y = -0.000001409530, 0.000000539837
        #         if led_arrays[index] == 13:
        #             diff_x , diff_y = -0.000001667030, 0.000004499406
        #         if led_arrays[index] == 12:
        #             diff_x , diff_y = 0.000002900621, -0.000004805893
        #         if led_arrays[index] == 11:
        #             diff_x , diff_y = 0.001056132598, 0.001399757881
        #
        #     if idx == 5:
        #         if led_arrays[index] == 14:
        #             diff_x , diff_y = 0.000000222007, -0.000000343465
        #         if led_arrays[index] == 13:
        #             diff_x , diff_y = -0.000000007044, 0.000000758135
        #         if led_arrays[index] == 12:
        #             diff_x , diff_y = -0.000000211343, -0.000000402673
        #         if led_arrays[index] == 11:
        #             diff_x , diff_y = -0.000087004477, -0.000051378941
        #
        #     if idx == 6:
        #         if led_arrays[index] == 3:
        #             diff_x, diff_y = 0.000000095450, -0.000000034728
        #         if led_arrays[index] == 4:
        #             diff_x, diff_y = 0.000000121489, -0.000000041803
        #         if led_arrays[index] == 5:
        #             diff_x, diff_y = -0.000000250158, 0.000000083608
        #         if led_arrays[index] == 6:
        #             diff_x, diff_y = 0.001059224329, 0.000822011599
        #
        #     if idx == 7:
        #         if led_arrays[index] == 3:
        #             diff_x, diff_y = -0.000000056492, 0.000000020008
        #         if led_arrays[index] == 4:
        #             diff_x, diff_y = -0.000000023278, -0.000000015492
        #         if led_arrays[index] == 5:
        #             diff_x, diff_y = 0.000000086156, -0.000000010166
        #         if led_arrays[index] == 6:
        #             diff_x, diff_y = -0.000090665508, 0.001059464543
        #
        #     if idx == 8:
        #         if led_arrays[index] == 6:
        #             diff_x, diff_y = 0.000000001313, 0.000000002243
        #         if led_arrays[index] == 7:
        #             diff_x, diff_y = 0.000000001834, 0.000000010734
        #         if led_arrays[index] == 8:
        #             diff_x, diff_y = 0.000000003025, 0.000000004124
        #         if led_arrays[index] == 9:
        #             diff_x, diff_y = -0.002831087529, 0.000090304859
        #
        #     if idx == 9:
        #         if led_arrays[index] == 6:
        #             diff_x, diff_y = -0.000000000377, 0.000000001492
        #         if led_arrays[index] == 7:
        #             diff_x, diff_y = -0.000000008057, 0.000000003257
        #         if led_arrays[index] == 8:
        #             diff_x, diff_y = 0.000000015187, 0.000000012256
        #         if led_arrays[index] == 9:
        #             diff_x, diff_y = 0.000264298114, 0.000401437889
        #
        #     real_list_2d_undistorted[index][0] -= float(diff_x)
        #     real_list_2d_undistorted[index][1] -= float(diff_y)



        # list_2d_undistorted_noise = []
        # for leds in list_2d_undistorted:
        #     list_2d_undistorted_noise.append([leds[0][0], leds[0][1]])

        ## test code pose offset test code #####
        #############################################################
        nret = cv2.fisheye.projectPoints(noise_leds, Rod, obj_cam_pos_n,
                                         cam_pose['cam_cal']['cameraK'],
                                         cam_pose['cam_cal']['dist_coeff'])
        n_xx, n_yy = nret[0].reshape(len(noise_leds), 2).transpose()
        blobs_2d_noise = []
        for i in range(len(n_xx)):
            blobs_2d_noise.append([[n_xx[i], n_yy[i]]])
        blobs_2d_noise = np.array(blobs_2d_noise, dtype=np.float64)

        blobs_2d_noise_undistort = cv2.fisheye.undistortPoints(blobs_2d_noise,
                                                               cam_pose['cam_cal']['cameraK'],
                                                               cam_pose['cam_cal']['dist_coeff'])
        cam_pose['test'] = blobs_2d_noise_undistort
        ###########################################################################################

        check = len(origin_leds)
        if check != len(blobs_2d):
            print("assertion not equal: ", len(blobs_2d))
            ax_ransac.set_title(f'assertion error={len(blobs_2d)}')
            continue

        if check < 4 or len(blobs_2d) < 4:
            print("assertion < 4: ", check)
            ax_ransac.set_title(f'assertion error={check}')
            continue

        # real
        # _, rrvecs, rtvecs, inliers = cv2.solvePnPRansac(origin_leds, real_list_2d_undistorted,
        #                                                 cameraK,
        #                                                 distCoeff,
        #                                                 useExtrinsicGuess=True,
        #                                                 iterationsCount=100,
        #                                                 confidence=0.99,
        #                                                 reprojectionError=8.0,
        #                                                 flags=cv2.SOLVEPNP_ITERATIVE)
        _, rrvecs, rtvecs, inliers = cv2.solvePnPRansac(origin_leds, real_list_2d_undistorted,
                                                        cameraK,
                                                        distCoeff,
                                                        flags=cv2.SOLVEPNP_AP3P)
        rRER = '%0.8f' % cal_RER_px(led_arrays,
                                    origin_leds,
                                    real_list_2d_undistorted,
                                    inliers,
                                    rrvecs,
                                    rtvecs,
                                    cameraK,
                                    distCoeff)

        # simulation
        # _, srvecs, stvecs, dinliers = cv2.solvePnPRansac(origin_leds, list_2d_undistorted,
        #                                                  cameraK,
        #                                                  distCoeff,
        #                                                  useExtrinsicGuess=True,
        #                                                  iterationsCount=100,
        #                                                  confidence=0.99,
        #                                                  reprojectionError=8.0,
        #                                                  flags=cv2.SOLVEPNP_ITERATIVE)
        _, srvecs, stvecs, dinliers = cv2.solvePnPRansac(origin_leds, list_2d_undistorted,
                                                         cameraK,
                                                         distCoeff,
                                                         flags=cv2.SOLVEPNP_AP3P)
        sRER = '%0.8f' % cal_RER_px(led_arrays,
                                    origin_leds,
                                    list_2d_undistorted,
                                    dinliers,
                                    srvecs,
                                    stvecs,
                                    cameraK,
                                    distCoeff)
        print('rRER: ', rRER, ' sRER: ', sRER)
        cam_pose['coord']['undistort_s'] = list_2d_undistorted
        cam_pose['coord']['distort_s'] = blobs_2d
        cam_pose['coord']['undistort_r'] = real_list_2d_undistorted
        cam_pose['coord']['distort_r'] = real_blobs_2d






        # print('cam_id: ', idx, ' : ', RER, ' ', dRER)
        # print('facing: ', led_arrays)
        # print('distort')
        # print('rvecs ', rvecs[:, 0], ' tvecs ', tvecs[:, 0])
        cam_pose['rt']['rvecs_r'] = rrvecs[:, 0]
        cam_pose['rt']['tvecs_r'] = rtvecs[:, 0]
        # print('undistort')
        # print('rvecs ', drvecs[:, 0], ' tvecs ', dtvecs[:, 0])
        cam_pose['rt']['rvecs_s'] = srvecs[:, 0]
        cam_pose['rt']['tvecs_s'] = stvecs[:, 0]

        # 여기서 pose apply 해보자
        #############
        observed_pos_r = rtvecs.reshape(3)
        observed_rot_r = R.from_rotvec(rrvecs.reshape(3)).as_quat()
        observed_pose_r = {'position': vector3(observed_pos_r[0], observed_pos_r[1], observed_pos_r[2]),
                           'orient': quat(observed_rot_r[0], observed_rot_r[1], observed_rot_r[2], observed_rot_r[3])}
        remake_offset_r = pose_apply_inverse(observed_pose_r, camera_array[idx])

        remake_leds_r = []
        for x in new_pts_facing:
            pt = vector3(x['pos'][0], x['pos'][1], x['pos'][2])
            temp_pt = transfer_point(pt, remake_offset_r)
            remake_leds_r.append([[temp_pt.x, temp_pt.y, temp_pt.z]])
        remake_leds_r = np.array(remake_leds_r, dtype=np.float64)
        tvec_r = np.array(remake_offset_r['position'])
        rot_r = R.from_quat(np.array(remake_offset_r['orient']))
        rvec_r, _ = cv2.Rodrigues(rot_r.as_matrix())
        cam_pose['rt']['r_rvecs_r'] = rvec_r
        cam_pose['rt']['r_tvecs_r'] = tvec_r

        ret = cv2.fisheye.projectPoints(remake_leds_r,
                                        cam_pose['S_R_T']['rvecs'],
                                        cam_pose['S_R_T']['tvecs'],
                                        cam_pose['cam_cal']['cameraK'],
                                        cam_pose['cam_cal']['dist_coeff'])

        xx, yy = ret[0].reshape(len(remake_leds_r), 2).transpose()

        blobs_2d_remake_r = []
        for i in range(len(xx)):
            blobs_2d_remake_r.append([[xx[i], yy[i]]])

        blobs_2d_remake_distort_r = np.array(blobs_2d_remake_r, dtype=np.float64)
        list_2d_remake_undistorted_r = cv2.fisheye.undistortPoints(blobs_2d_remake_distort_r,
                                                                   cam_pose['cam_cal']['cameraK'],
                                                                   cam_pose['cam_cal']['dist_coeff'])
        cam_pose['coord']['remake_undistort_r'] = list_2d_remake_undistorted_r[:, 0]

        #############
        observed_pos_s = stvecs.reshape(3)
        observed_rot_s = R.from_rotvec(srvecs.reshape(3)).as_quat()
        observed_pose_s = {'position': vector3(observed_pos_s[0], observed_pos_s[1], observed_pos_s[2]),
                           'orient': quat(observed_rot_s[0], observed_rot_s[1], observed_rot_s[2], observed_rot_s[3])}
        remake_offset_s = pose_apply_inverse(observed_pose_s, camera_array[idx])

        remake_leds_s = []
        for x in new_pts_facing:
            pt = vector3(x['pos'][0], x['pos'][1], x['pos'][2])
            temp_pt = transfer_point(pt, remake_offset_s)
            remake_leds_s.append([[temp_pt.x, temp_pt.y, temp_pt.z]])
        remake_leds_s = np.array(remake_leds_s, dtype=np.float64)
        tvec_s = np.array(remake_offset_s['position'])
        rot_s = R.from_quat(np.array(remake_offset_s['orient']))
        rvec_s, _ = cv2.Rodrigues(rot_s.as_matrix())
        cam_pose['rt']['r_rvecs_s'] = rvec_s
        cam_pose['rt']['r_tvecs_s'] = tvec_s

        ret = cv2.fisheye.projectPoints(remake_leds_s,
                                        cam_pose['S_R_T']['rvecs'],
                                        cam_pose['S_R_T']['tvecs'],
                                        cam_pose['cam_cal']['cameraK'],
                                        cam_pose['cam_cal']['dist_coeff'])

        xx, yy = ret[0].reshape(len(remake_leds_s), 2).transpose()

        blobs_2d_remake_s = []
        for i in range(len(xx)):
            blobs_2d_remake_s.append([[xx[i], yy[i]]])

        blobs_2d_remake_distort_s = np.array(blobs_2d_remake_s, dtype=np.float64)
        list_2d_remake_undistorted_s = cv2.fisheye.undistortPoints(blobs_2d_remake_distort_s,
                                                                   cam_pose['cam_cal']['cameraK'],
                                                                   cam_pose['cam_cal']['dist_coeff'])
        cam_pose['coord']['remake_undistort_s'] = list_2d_remake_undistorted_s[:, 0]

        # print('led ', cam_pose['model']['leds'])
        # print('real')
        # print('d ', cam_pose['distorted_2d'])
        # print('u ', cam_pose['undistorted_2d'])
        #
        # print('simulation')
        # print('led nums : ', led_arrays)
        # print('d ', blobs_2d[:, 0])
        # print('u ', list_2d_undistorted[:, 0])
        #
        # print('simulation')
        # print('led nums : ', led_arrays)
        # print('rr ', list_2d_remake_undistorted_r[:, 0])
        # print('rs ', list_2d_remake_undistorted_s[:, 0])

        # ax_ransac.set_title(f'[{idx}]:'f'{RER}', fontdict={'fontsize': 8, 'fontweight': 'medium'})
        # RER = float(RER)
        # dRER = float(dRER)
        # origin을 구한 rt로 다시 투영
        points2D_reproj = cv2.fisheye.projectPoints(origin_leds, srvecs, stvecs,
                                                    cam_pose['cam_cal']['cameraK'],
                                                    cam_pose['cam_cal']['dist_coeff'])
        r_xx, r_yy = points2D_reproj[0].reshape(len(origin_leds), 2).transpose()

        origin_pts = []
        noise_pts = []
        for i in range(len(r_xx)):
            _opos = list(map(float, [r_xx[i], r_yy[i]]))
            origin_pts.append({'idx': led_arrays[i], 'pos': _opos, 'reserved': 0})
            _pos = list(map(float, [n_xx[i], n_yy[i]]))
            noise_pts.append({'idx': led_arrays[i], 'pos': _pos, 'reserved': 0})

        draw_dots(2, origin_pts, ax_ransac, 'red')
        draw_dots(2, noise_pts, ax_ransac, 'blue')

        max_x = 0
        max_y = 0
        if abs(max(n_xx)) > max_x: max_x = abs(max(n_xx))
        if abs(min(n_xx)) > max_x: max_x = abs(min(n_xx))
        if abs(max(n_yy)) > max_y: max_y = abs(max(n_yy))
        if abs(min(n_yy)) > max_y: max_y = abs(min(n_yy))
        dimen = max(max_x, max_y)
        dimen *= 1.1
        ax_ransac.set_xlim([500, dimen])
        ax_ransac.set_ylim([300, dimen])
        #
        #     if RER != -1:
        #         rer_array.append(RER)
        #         cam_id.append(f'{idx}')
        #         if RER > 1.0:
        #             cnt += 1
        #         else:
        #             normal_rer.append(RER)
        #             default_rer.append(dRER)
        #
        # rer_tmp = 0
        # drer_tmp = 0
        # for rer in normal_rer:
        #     rer_tmp += rer
        #
        # for rer in default_rer:
        #     drer_tmp += rer
        #
        # if len(normal_rer) > 0 and len(default_rer) > 0:
        #     avg_rer = rer_tmp / len(normal_rer)
        #     davg_rer = drer_tmp / len(default_rer)
        #     print(title, ' avg: ', '%0.8f' % avg_rer, ':', '%0.8f' % davg_rer, 'over cnt: ', cnt)

        # plt.subplots_adjust(hspace=0.2, wspace=0.2)
        #
        # plt.style.use('default')
        # plt.figure(figsize=(15, 10))
        # plt.title('RER')
        # markers, stemlines, baseline = plt.stem(cam_id, rer_array)
        # markers.set_color('red')

        print('##############################\n')


def pair_points(target):
    for cam_id, data in enumerate(camera_array):
        # print('cam_id ', cam_id)
        # print(data)
        for i in range(len(data['pts_facing']['real']) - 1):
            facing_dot = data['pts_facing']['real'][i]
            led_num = int(facing_dot['idx'])
            # print(led_num)
            if len(data['coord']['undistort_r']) <= 0 or len(data['coord']['undistort_s']) <= 0:
                continue

            r_undistorted_x = data['coord']['undistort_r'][i][0]
            r_undistorted_y = data['coord']['undistort_r'][i][1]
            s_undistorted_x = data['coord']['undistort_s'][i][0]
            s_undistorted_y = data['coord']['undistort_s'][i][1]
            r_s_undistorted_x = data['coord']['remake_undistort_s'][i][0]
            r_s_undistorted_y = data['coord']['remake_undistort_s'][i][1]
            r_r_undistorted_x = data['coord']['remake_undistort_r'][i][0]
            r_r_undistorted_y = data['coord']['remake_undistort_r'][i][1]
            leds_data[target][led_num]['pair_xy'].append({'cidx': cam_id,
                                                          'led_num': led_num,
                                                          'rcx': r_undistorted_x,
                                                          'rcy': r_undistorted_y,
                                                          'scx': s_undistorted_x,
                                                          'scy': s_undistorted_y,
                                                          'rscx': r_s_undistorted_x,
                                                          'rscy': r_s_undistorted_y,
                                                          'rrcx': r_r_undistorted_x,
                                                          'rrcy': r_r_undistorted_y,
                                                          })

    print('pair_xy result and remake 3d')
    for i in range(len(leds_data[target])):
        # if len(leds_data[target][i]['pair_xy']) > 0:
        #     print('led_num ', i, ' ', leds_data[target][i]['pair_xy'])

        led_pair_cnt = len(leds_data[target][i]['pair_xy'])
        if led_pair_cnt < 2:
            print(f'Error LED Num {i} has no more than 2-cameras')
            leds_data[target][i]['remake_3d'] = 'error'
        else:
            comb_led = list(itertools.combinations(leds_data[target][i]['pair_xy'], 2))
            for data in comb_led:
                # print('comb_led idx: ', i, ' : ', data)
                undistort_r_1 = coord_refine(camera_array, data[0], data[1], None, None, 'rcx', 'rcy', True)
                undistort_r_2 = coord_refine(camera_array, data[0], data[1], 'rvecs_r', 'tvecs_r', 'rcx', 'rcy', False)
                undistort_s = coord_refine(camera_array, data[0], data[1], 'rvecs_s', 'tvecs_s', 'scx', 'scy', False)

                undistort_r_s = coord_refine(camera_array, data[0], data[1], 'rvecs_s', 'tvecs_s', 'rscx', 'rscy', False)
                undistort_r_r = coord_refine(camera_array, data[0], data[1], 'rvecs_r', 'tvecs_r', 'rrcx', 'rrcy', False)

                undistort_r_s_2 = coord_refine(camera_array, data[0], data[1], 'r_rvecs_s', 'r_tvecs_s', 'scx', 'scy', False)
                undistort_r_r_2 = coord_refine(camera_array, data[0], data[1], 'r_rvecs_r', 'r_tvecs_r', 'rcx', 'rcy', False)

                # print(undistort_r, ' vs ', undistort_s)
                leds_data[target][i]['remake_3d'].append(
                    {'cam_l': data[0]['cidx'], 'cam_r': data[1]['cidx'],
                     # STATIC RT
                     'uncoord_r_1': undistort_r_1,
                     # Dynamic RT
                     'uncoord_r_2': undistort_r_2,

                     # Pose Apply to Simulation
                     'undistort_r_s': undistort_r_s,
                     # Pose Applyt to Dynamic RT
                     'undistort_r_r': undistort_r_r,

                     # Projection Point (Simulation)
                     'uncoord_s': undistort_s,
                     'uncoord_r_s_2': undistort_r_s_2,
                     'uncoord_r_r_2': undistort_r_r_2,
                     })


def coord_refine(cam_array, cam_l, cam_r, rvec, tvec, cx, cy, status):
    cam_l_id = cam_l['cidx']
    cam_r_id = cam_r['cidx']

    if status:
        l_rvec = cam_array[cam_l_id]['S_R_T']['rvecs']
        r_rvec = cam_array[cam_r_id]['S_R_T']['rvecs']
        l_tvec = cam_array[cam_l_id]['S_R_T']['tvecs']
        r_tvec = cam_array[cam_r_id]['S_R_T']['tvecs']
    else:
        l_rvec = cam_array[cam_l_id]['rt'][rvec]
        r_rvec = cam_array[cam_r_id]['rt'][rvec]
        l_tvec = cam_array[cam_l_id]['rt'][tvec]
        r_tvec = cam_array[cam_r_id]['rt'][tvec]

    left_rotation, jacobian = cv2.Rodrigues(l_rvec)
    right_rotation, jacobian = cv2.Rodrigues(r_rvec)

    # projection matrices:
    RT = np.zeros((3, 4))
    RT[:3, :3] = left_rotation
    RT[:3, 3] = l_tvec.transpose()
    left_projection = np.dot(cameraK, RT)

    RT = np.zeros((3, 4))
    RT[:3, :3] = right_rotation
    RT[:3, 3] = r_tvec.transpose()
    right_projection = np.dot(cameraK, RT)

    # print('cam_l ', cam_l[cx], ' ', cam_l[cy])
    # print('cam_r ', cam_r[cx], ' ', cam_r[cy])
    # print('left_projection ', left_projection)
    # print('right projection ', right_projection)
    triangulation = cv2.triangulatePoints(left_projection, right_projection,
                                          (cam_l[cx], cam_l[cy]),
                                          (cam_r[cx], cam_r[cy]))
    homog_points = triangulation.transpose()

    get_points = cv2.convertPointsFromHomogeneous(homog_points)

    return get_points


def print_result(origin, target):
    print('remake 3d result')
    led_num_o_t = []
    undistort_distance_r_1 = []
    undistort_distance_r_2 = []
    undistort_distance_s = []
    undistort_distance_o_t = []
    remake_distance_s = []
    remake_distance_r = []

    remake_distance_s_2 = []
    remake_distance_r_2 = []

    # for lsm
    Pc = []
    led_idx = []
    origin_pts = []
    before_pts = []
    after_pts = []

    for i, data in enumerate(leds_data[target]):
        if data['remake_3d'] != 'error':
            for remake_data in data['remake_3d']:
                origin_x = leds_data[origin][i]['pos'][0]
                origin_y = leds_data[origin][i]['pos'][1]
                origin_z = leds_data[origin][i]['pos'][2]
                target_x = leds_data[target][i]['pos'][0]
                target_y = leds_data[target][i]['pos'][1]
                target_z = leds_data[target][i]['pos'][2]

                diff_x = '%0.12f' % (origin_x - target_x)
                diff_y = '%0.12f' % (origin_y - target_y)
                diff_z = '%0.12f' % (origin_z - target_z)
                # print('o_t:[', diff_x, ',', diff_y, ',', diff_z, ']')
                distance = np.sqrt(
                    np.power(float(diff_x), 2) + np.power(float(diff_y), 2) + np.power(
                        float(diff_z), 2))
                undistort_distance_o_t.append(distance)
                led_num_o_t.append(f'LED {i}')

                diff_x_r = '%0.12f' % (target_x - remake_data['uncoord_r_1'][0][0][0])
                diff_y_r = '%0.12f' % (target_y - remake_data['uncoord_r_1'][0][0][1])
                diff_z_r = '%0.12f' % (target_z - remake_data['uncoord_r_1'][0][0][2])
                # print('real:[', diff_x_r, ',', diff_y_r, ',', diff_z_r, ']')
                r_distance_1 = np.sqrt(
                    np.power(float(diff_x_r), 2) + np.power(float(diff_y_r), 2) + np.power(
                        float(diff_z_r), 2))
                undistort_distance_r_1.append(r_distance_1)

                diff_x_r = '%0.12f' % (target_x - remake_data['uncoord_r_2'][0][0][0])
                diff_y_r = '%0.12f' % (target_y - remake_data['uncoord_r_2'][0][0][1])
                diff_z_r = '%0.12f' % (target_z - remake_data['uncoord_r_2'][0][0][2])
                # print('real:[', diff_x_r, ',', diff_y_r, ',', diff_z_r, ']')
                r_distance_2 = np.sqrt(
                    np.power(float(diff_x_r), 2) + np.power(float(diff_y_r), 2) + np.power(
                        float(diff_z_r), 2))
                undistort_distance_r_2.append(r_distance_2)

                diff_x_s = '%0.12f' % (target_x - remake_data['uncoord_s'][0][0][0])
                diff_y_s = '%0.12f' % (target_y - remake_data['uncoord_s'][0][0][1])
                diff_z_s = '%0.12f' % (target_z - remake_data['uncoord_s'][0][0][2])
                # print('simulation:[', diff_x_s, ',', diff_y_s, ',', diff_z_s, ']')
                s_distance = np.sqrt(
                    np.power(float(diff_x_s), 2) + np.power(float(diff_y_s), 2) + np.power(
                        float(diff_z_s), 2))
                undistort_distance_s.append(s_distance)

                diff_x = '%0.12f' % (target_x - remake_data['undistort_r_s'][0][0][0])
                diff_y = '%0.12f' % (target_y - remake_data['undistort_r_s'][0][0][1])
                diff_z = '%0.12f' % (target_z - remake_data['undistort_r_s'][0][0][2])
                # print('remake_s:[', diff_x, ',', diff_y, ',', diff_z, ']')
                distance = np.sqrt(
                    np.power(float(diff_x), 2) + np.power(float(diff_y), 2) + np.power(
                        float(diff_z), 2))
                remake_distance_s.append(distance)

                #
                diff_x = '%0.12f' % (target_x - remake_data['undistort_r_r'][0][0][0])
                diff_y = '%0.12f' % (target_y - remake_data['undistort_r_r'][0][0][1])
                diff_z = '%0.12f' % (target_z - remake_data['undistort_r_r'][0][0][2])
                # print('remake_r:[', diff_x, ',', diff_y, ',', diff_z, ']')
                distance = np.sqrt(
                    np.power(float(diff_x), 2) + np.power(float(diff_y), 2) + np.power(
                        float(diff_z), 2))
                remake_distance_r.append(distance)

                diff_x = '%0.12f' % (target_x - remake_data['uncoord_r_s_2'][0][0][0])
                diff_y = '%0.12f' % (target_y - remake_data['uncoord_r_s_2'][0][0][1])
                diff_z = '%0.12f' % (target_z - remake_data['uncoord_r_s_2'][0][0][2])
                # print('remake_r:[', diff_x, ',', diff_y, ',', diff_z, ']')
                distance = np.sqrt(
                    np.power(float(diff_x), 2) + np.power(float(diff_y), 2) + np.power(
                        float(diff_z), 2))
                remake_distance_s_2.append(distance)

                diff_x = '%0.12f' % (target_x - remake_data['uncoord_r_r_2'][0][0][0])
                diff_y = '%0.12f' % (target_y - remake_data['uncoord_r_r_2'][0][0][1])
                diff_z = '%0.12f' % (target_z - remake_data['uncoord_r_r_2'][0][0][2])
                # print('remake_r:[', diff_x, ',', diff_y, ',', diff_z, ']')
                distance = np.sqrt(
                    np.power(float(diff_x), 2) + np.power(float(diff_y), 2) + np.power(
                        float(diff_z), 2))
                remake_distance_r_2.append(distance)

                # cal_diff_x = origin_x - remake_data['uncoord_s'][0][0][0]
                # cal_diff_y = origin_y - remake_data['uncoord_s'][0][0][1]
                # cal_diff_z = origin_z - remake_data['uncoord_s'][0][0][2]
                #
                new_x = remake_data['uncoord_r_1'][0][0][0]
                new_y = remake_data['uncoord_r_1'][0][0][1]
                new_z = remake_data['uncoord_r_1'][0][0][2]
                # print(cal_diff_x, ' ', cal_diff_y, ' ', cal_diff_z)

                before_pts.append({'idx': i,
                                   'pos': [new_x,
                                           new_y,
                                           new_z]})
                Pc.append([new_x,
                           new_y,
                           new_z])
                origin_pts.append(leds_data[target][i]['pos'])

    plt.style.use('default')
    plt.figure(figsize=(15, 10))
    plt.title('Diff result')
    markers, stemlines, baseline = plt.stem(led_num_o_t, undistort_distance_r_1, label='real_1(STATIC)')
    markers.set_color('red')
    markers, stemlines, baseline = plt.stem(led_num_o_t, undistort_distance_r_2, label='real_2(DYNAMIC)')
    markers.set_color('blue')
    markers, stemlines, baseline = plt.stem(led_num_o_t, undistort_distance_s, label='simulation')
    markers.set_color('black')
    markers, stemlines, baseline = plt.stem(led_num_o_t, undistort_distance_o_t, label='oigin vs target')
    markers.set_color('green')
    markers, stemlines, baseline = plt.stem(led_num_o_t, remake_distance_r, label='remake to DYNAMIC')
    markers.set_color('purple')
    markers, stemlines, baseline = plt.stem(led_num_o_t, remake_distance_s, label='remake to Simulation')
    markers.set_color('magenta')
    #
    # markers, stemlines, baseline = plt.stem(led_num_o_t, remake_distance_r_2, label='remake_distance_r_2')
    # markers.set_color('yellow')
    # markers, stemlines, baseline = plt.stem(led_num_o_t, remake_distance_s_2, label='remake_distance_s_2')
    # markers.set_color('pink')

    # for i, data in enumerate(leds_data[target]):
    #     print('led ', i)
    #     # print(data['min'])
    #     cam_l_id = data['min']['remake']['cam'][0]
    #     cam_r_id = data['min']['remake']['cam'][1]
    #     print(cam_l_id, ' ', cam_r_id)
    #     print('cam_l\n', camera_array[cam_l_id]['pts_facing']['real'])
    #     print('cam_r\n', camera_array[cam_r_id]['pts_facing']['real'])
    #
    #     for ilx, led_i in enumerate(camera_array[cam_l_id]['pts_facing']['real']):
    #         l_idx = int(led_i['idx'])
    #         if i == l_idx:
    #             cam_l_led_num = ilx
    #             break
    #
    #     for irx, led_i in enumerate(camera_array[cam_r_id]['pts_facing']['real']):
    #         r_idx = int(led_i['idx'])
    #         if i == r_idx:
    #             cam_r_led_num = irx
    #             break
    #
    #     left_rotation, jacobian = cv2.Rodrigues(camera_array[cam_l_id]['urvecs'])
    #     right_rotation, jacobian = cv2.Rodrigues(camera_array[cam_r_id]['urvecs'])
    #
    #     # projection matrices:
    #     RT = np.zeros((3, 4))
    #     RT[:3, :3] = left_rotation
    #     RT[:3, 3] = camera_array[cam_l_id]['utvecs'].transpose()
    #     left_projection = np.dot(cameraK, RT)
    #
    #     RT = np.zeros((3, 4))
    #     RT[:3, :3] = right_rotation
    #     RT[:3, 3] = camera_array[cam_r_id]['utvecs'].transpose()
    #     right_projection = np.dot(cameraK, RT)
    #     print(camera_array[cam_l_id]['test'])
    #     print(camera_array[cam_r_id]['test'])
    #     lcx = camera_array[cam_l_id]['test'][cam_l_led_num][0][0]
    #     lcy = camera_array[cam_l_id]['test'][cam_l_led_num][0][1]
    #     rcx = camera_array[cam_r_id]['test'][cam_r_led_num][0][0]
    #     rcy = camera_array[cam_r_id]['test'][cam_r_led_num][0][1]
    #     print(lcx, ' ', lcy, ' ', rcx, ' ', rcy)
    #     # print(camera_array[cam_r_id]['test'])
    #     triangulation = cv2.triangulatePoints(left_projection, right_projection,
    #                                           (lcx, lcy),
    #                                           (rcx, rcy))
    #     homog_points = triangulation.transpose()
    #
    #     get_points = cv2.convertPointsFromHomogeneous(homog_points)
    #
    #     print('result ', data['pos'])
    #     print(get_points)
    #
    #     before_pts.append({'idx': i,
    #                        'pos': get_points[0][0]})
    #     Pc.append(get_points[0][0])
    #     origin_pts.append(leds_data[origin][i]['pos'])

    for data in Pc:
        print(data)

    A = np.vstack([np.array(Pc).T, np.ones(len(Pc))]).T
    Rt = np.linalg.lstsq(A, np.array(origin_pts), rcond=None)[0]
    print("Rt=\n", Rt.T)
    # Pd_est : Pa 예측값 (Calibration 값)
    Pd_est = np.matmul(Rt.T, A.T)

    print("Pd_est=\n", Pd_est)
    Err = np.array(origin_pts) - Pd_est.T
    print("Err=\n", Err)

    for led_num in range(LEDS_COUNT):
        after_pts.append({'idx': led_num,
                          'pos': Pd_est.T[led_num]})

    # print('origin_pts: ', origin_pts)
    # print('before pts: ', before_pts)
    # print('after pts: ', after_pts)

    led_num = []
    distance_array = []
    for i, data in enumerate(leds_data[target]):
        if data['remake_3d'] != 'error':
            target_x = leds_data[target][i]['pos'][0]
            target_y = leds_data[target][i]['pos'][1]
            target_z = leds_data[target][i]['pos'][2]

            diff_x = '%0.12f' % (target_x - after_pts[i]['pos'][0])
            diff_y = '%0.12f' % (target_y - after_pts[i]['pos'][1])
            diff_z = '%0.12f' % (target_z - after_pts[i]['pos'][2])
            print('lsm:[', diff_x, ',', diff_y, ',', diff_z, ']')
            distance = np.sqrt(
                np.power(float(diff_x), 2) + np.power(float(diff_y), 2) + np.power(
                    float(diff_z), 2))
            distance_array.append(distance)
            led_num.append(f'LED {i}')
    markers, stemlines, baseline = plt.stem(led_num, distance_array, label='LSM')
    markers.set_color('gray')

    plt.legend()
    #
    # RVECS = np.array([[rvecs[0]], [rvecs[1]], [rvecs[2]]], dtype=np.float64)
    # TVECS = np.array([[tvecs[0]], [tvecs[1]], [tvecs[2]]], dtype=np.float64)
    #
    # print('rvecs: ', RVECS)
    # print('tvecs: ', TVECS)

    # cam_pos = TVECS.reshape(3)
    # cam_ori = R.from_rotvec(RVECS.reshape(3)).as_quat()

    #
    # print('pos: ', cam_pos, ' ori(euler): ', cam_ori_euler,
    #       ' ori(quat): ', cam_ori,
    #       ' ori(re-quat): ', cam_ori_quat)
    #
    # if i % 2 == 0:
    #     group_num = i
    #
    # camera_info_array.append(
    #     {'cidx': i,
    #      'position': vector3(cam_pos[0], cam_pos[1], cam_pos[2]),
    #      'orient': quat(cam_ori[0], cam_ori[1], cam_ori[2], cam_ori[3]),
    #
    # remake_leds_r = []
    # for x in new_pts_facing:
    #     pt = vector3(x['pos'][0], x['pos'][1], x['pos'][2])
    #     temp_pt = transfer_point(pt, remake_offset_r)
    #     remake_leds_r.append([[temp_pt.x, temp_pt.y, temp_pt.z]])
    # remake_leds_r = np.array(remake_leds_r, dtype=np.float64)

    # for idx, cam_pose in enumerate(camera_array):
    #     for i in range(len(data['pts_facing']['real']) - 1):
    #         facing_dot = data['pts_facing']['real'][i]
    #         led_num = int(facing_dot['idx'])
    #         leds_data[target][led_num]

    # for i, data in enumerate(leds_data[target]):
    #     if data['remake_3d'] != 'error':
    #         for remake_data in data['remake_3d']:
    #             print('1:', remake_data['uncoord_r_1'])
    #             RVECS = []
    #             TVECS = []
    #             RVECS = np.array([[cam_pos['rt']['r_rvecs_s'][0]], [cam_pos['rt']['r_rvecs_s'][1]], [cam_pos['rt']['r_rvecs_s'][2]]], dtype=np.float64)
    #             TVECS = np.array([[cam_pos['rt']['t_rvecs_s'][0]], [cam_pos['rt']['t_rvecs_s'][1]], [cam_pos['rt']['t_rvecs_s'][2]]], dtype=np.float64)
    #             cam_pos = TVECS.reshape(3)
    #             cam_ori = R.from_rotvec(RVECS.reshape(3)).as_quat()
    #             remake_offset = {'position': vector3(cam_pos[0], cam_pos[1], cam_pos[2]),
    #                              'orient': quat(cam_ori[0], cam_ori[1], cam_ori[2], cam_ori[3])}
    #             pt = vector3(remake_data['uncoord_r_1'][0][0][0], remake_data['uncoord_r_1'][0][0][1], remake_data['uncoord_r_1'][0][0][2])
    #             temp_pt = transfer_point(pt, remake_offset)
    #             print('2: ', temp_pt)


def add_random():
    avg = 0
    sd = 0.001
    offset = np.random.normal(avg, sd, 3)
    for i, v in enumerate(offset):
        offset[i] = np.round_(np.clip(v, -sd, sd), 6)
    return offset


if __name__ == "__main__":
    print('start ransac test main')
    plt.style.use('default')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    path_1 = "../cal_uvc/jsons/"
    path_2 = '../cal_uvc/jsons/backup/good_6_2/'
    path_3 = '../cal_uvc/jsons/backup/static_6/'
    path_4 = '../cal_uvc/jsons/backup/static_6_2/'
    path_5 = '../cal_uvc/jsons/backup/static_6_2_disable_refine/'
    camera_array = init_model_json(path_4)

    origin = 'rift_6'
    target = 'rift_2'

    # read origin data
    read_led_pts(origin)
    # read target data
    read_led_pts(target)

    # make noise + pose added data
    # leds_data[target] = []
    # for led in leds_data[origin]:
    #     random_offset = add_random()
    #     idx = led['idx']
    #     x = led['pos'][0] + float(random_offset[0])
    #     y = led['pos'][1] + float(random_offset[1])
    #     z = led['pos'][2] + float(random_offset[2])
    #
    #     pt = transfer_point(vector3(x, y, z), pos_offset)
    #     leds_data[target].append({'idx': idx, 'pos': [pt.x, pt.y, pt.z], 'dir': led['dir'],
    #                               'pair_xy': [],
    #                               'remake_3d': [],
    #                               'min': {'dis': 10, 'remake': []}})

    print('origin')
    for led in leds_data[origin]:
        print(led)

    print('target')
    for led in leds_data[target]:
        print(led)

    facing_dot_area(origin)
    # ransac_test(origin, target)
    # #
    # pair_points(target)
    # print_result(origin, target)
    #
    # draw_blobs(ax)
    plt.show()
