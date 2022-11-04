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
from ransac_test import *


#
# def solvepnp_test(origin, target):
#     interationsCount = 100
#     confidence = 0.99
#
#     for cam_id, camera_info in enumerate(camera_array_origin):
#         print('cam_id ', cam_id)
#         print('leds ', camera_info['model']['leds'])
#
#         origin_leds = []
#         target_leds = []
#         camera_info['pts_facing']['real'] = camera_info['model']['leds']
#         for led_num in camera_info['model']['leds']:
#             origin_leds.append([[leds_data[origin][led_num]['pos'][0],
#                                  leds_data[origin][led_num]['pos'][1],
#                                  leds_data[origin][led_num]['pos'][2]]])
#             target_leds.append([[leds_data[target][led_num]['pos'][0],
#                                  leds_data[target][led_num]['pos'][1],
#                                  leds_data[target][led_num]['pos'][2]]])
#
#         origin_leds = np.array(origin_leds, dtype=np.float64)
#         target_leds = np.array(target_leds, dtype=np.float64)
#
#         list_2d_undistorted_origin = np.array(camera_info['undistorted_2d'], dtype=np.float64)
#         print('origin 2d undistort ', list_2d_undistorted_origin)
#         list_2d_undistorted_target = np.array(camera_array_target[cam_id]['undistorted_2d'], dtype=np.float64)
#         print('target 2d undistort ', list_2d_undistorted_target)
#
#         _, orvecs, otvecs, inliers = cv2.solvePnPRansac(origin_leds, list_2d_undistorted_origin,
#                                                         cameraK,
#                                                         distCoeff,
#                                                         useExtrinsicGuess=True,
#                                                         iterationsCount=interationsCount,
#                                                         confidence=confidence,
#                                                         reprojectionError=1.0,
#                                                         flags=cv2.SOLVEPNP_ITERATIVE
#                                                         )
#
#         # print('before LM ransac orvecs ', orvecs[:, 0], ' otvecs ', otvecs[:, 0])
#         # cv2.solvePnPRefineLM(origin_leds,
#         #                      list_2d_undistorted_origin,
#         #                      cameraK, distCoeff,
#         #                      orvecs, otvecs)
#         # print('after LM ransac orvecs ', orvecs[:, 0], ' otvecs ', otvecs[:, 0])
#         # print('after origin 2d undistort ', list_2d_undistorted_origin)
#         _, trvecs, ttvecs, inliers = cv2.solvePnPRansac(origin_leds, list_2d_undistorted_target,
#                                                         cameraK,
#                                                         distCoeff,
#                                                         useExtrinsicGuess=True,
#                                                         iterationsCount=interationsCount,
#                                                         confidence=confidence,
#                                                         reprojectionError=1.0,
#                                                         flags=cv2.SOLVEPNP_ITERATIVE
#                                                         )
#         # print('before LM ransac trvecs ', trvecs[:, 0], ' ttvecs ', ttvecs[:, 0])
#         # cv2.solvePnPRefineLM(origin_leds,
#         #                      list_2d_undistorted_target,
#         #                      cameraK, distCoeff,
#         #                      trvecs, ttvecs)
#         # print('after LM ransac trvecs ', trvecs[:, 0], ' ttvecs ', ttvecs[:, 0])
#         # print('after target 2d undistort ', list_2d_undistorted_target)
#
#         re_offset = {'position': vector3(0.001, 0.001, 0.001), 'orient': get_quat_from_euler('xyz', [10, 10, 10])}
#         remake_target = []
#         for blob in target_leds:
#             random_offset = add_random()
#             x = blob[0][0]
#             y = blob[0][1]
#             z = blob[0][2]
#
#             # add random noise
#             x += float(random_offset[0])
#             y += float(random_offset[1])
#             z += float(random_offset[2])
#
#             pt = vector3(x, y, z)
#             temp_pt = transfer_point(pt, re_offset)
#             remake_target.append([[temp_pt.x, temp_pt.y, temp_pt.z]])
#
#         remake_target = np.array(remake_target, dtype=np.float64)
#
#         # remake TEST
#         ret = cv2.fisheye.projectPoints(remake_target,
#                                         camera_info['S_R_T']['rvecs'],
#                                         camera_info['S_R_T']['tvecs'],
#                                         camera_info['cam_cal']['cameraK'],
#                                         camera_info['cam_cal']['dist_coeff'])
#
#         xx, yy = ret[0].reshape(len(remake_target), 2).transpose()
#
#         blobs_2d_remake_s = []
#         for i in range(len(xx)):
#             blobs_2d_remake_s.append([[xx[i], yy[i]]])
#
#         blobs_2d_remake_distort_s = np.array(blobs_2d_remake_s, dtype=np.float64)
#         list_2d_remake_undistorted_s = cv2.fisheye.undistortPoints(blobs_2d_remake_distort_s,
#                                                                    camera_info['cam_cal']['cameraK'],
#                                                                    camera_info['cam_cal']['dist_coeff'])
#
#         _, rrvecs, rtvecs, inliers = cv2.solvePnPRansac(origin_leds, list_2d_remake_undistorted_s,
#                                                         cameraK,
#                                                         distCoeff,
#                                                         useExtrinsicGuess=True,
#                                                         iterationsCount=interationsCount,
#                                                         confidence=confidence,
#                                                         reprojectionError=1.0,
#                                                         flags=cv2.SOLVEPNP_ITERATIVE
#                                                         )
#
#         camera_info['coord']['remake_undistort_s'] = list_2d_remake_undistorted_s[:, 0]
#         camera_info['rt']['r_rvecs_s'] = rrvecs[:, 0]
#         camera_info['rt']['r_tvecs_s'] = rtvecs[:, 0]
#
#         # origin
#         camera_info['coord']['undistort_s'] = list_2d_undistorted_origin
#         camera_info['rt']['rvecs_s'] = orvecs[:, 0]
#         camera_info['rt']['tvecs_s'] = otvecs[:, 0]
#
#         # target
#         camera_info['coord']['undistort_r'] = list_2d_undistorted_target
#         camera_info['rt']['rvecs_r'] = trvecs[:, 0]
#         camera_info['rt']['tvecs_r'] = ttvecs[:, 0]
#
#     for cam_id, data in enumerate(camera_array_origin):
#         # print('cam_id ', cam_id)
#         # print(data)
#         for i in range(len(data['pts_facing']['real']) - 1):
#             facing_dot = data['pts_facing']['real'][i]
#             # print(facing_dot)
#             if len(data['coord']['undistort_r']) <= 0 or len(data['coord']['undistort_s']) <= 0:
#                 continue
#
#             leds_data[target][facing_dot]['pair_xy'].append({'cidx': cam_id,
#                                                              'led_num': facing_dot,
#                                                              'scx': data['coord']['undistort_s'][i][0],
#                                                              'scy': data['coord']['undistort_s'][i][1],
#
#                                                              'rcx': data['coord']['undistort_r'][i][0],
#                                                              'rcy': data['coord']['undistort_r'][i][1],
#
#                                                              'rscx': data['coord']['remake_undistort_s'][i][0],
#                                                              'rscy': data['coord']['remake_undistort_s'][i][1],
#                                                              })
#
#     print('pair_xy result and remake 3d')
#     for i in range(len(leds_data[target])):
#         # if len(leds_data[target][i]['pair_xy']) > 0:
#         #     print('led_num ', i, ' ', leds_data[target][i]['pair_xy'])
#
#         led_pair_cnt = len(leds_data[target][i]['pair_xy'])
#         if led_pair_cnt < 2:
#             print(f'Error LED Num {i} has no more than 2-cameras')
#             leds_data[target][i]['remake_3d'] = 'error'
#         else:
#             comb_led = list(itertools.combinations(leds_data[target][i]['pair_xy'], 2))
#             for data in comb_led:
#                 # print('comb_led idx: ', i, ' : ', data)
#                 undistort_s = coord_refine(camera_array_origin, data[0], data[1], 'rvecs_s', 'tvecs_s', 'scx', 'scy',
#                                            False)
#                 undistort_r = coord_refine(camera_array_origin, data[0], data[1], 'rvecs_r', 'tvecs_r', 'rcx', 'rcy',
#                                            False)
#                 r_undistort_s = coord_refine(camera_array_origin, data[0], data[1], 'r_rvecs_s', 'r_tvecs_s', 'rscx',
#                                              'rscy',
#                                              False)
#
#                 leds_data[target][i]['remake_3d'].append(
#                     {'cam_l': data[0]['cidx'], 'cam_r': data[1]['cidx'],
#                      'uncoord_s': undistort_s,
#                      'uncoord_r': undistort_r,
#                      'r_uncoord_s': r_undistort_s,
#                      })
#
#     print('remake 3d result')
#     led_num_o_t = []
#     undistort_distance_r = []
#     undistort_distance_s = []
#     undistort_distance_o_t = []
#     r_undistort_distance_s = []
#
#     # for lsm
#     Pc = []
#     origin_pts = []
#     before_pts = []
#     after_pts = []
#
#     for i, data in enumerate(leds_data[target]):
#         if data['remake_3d'] != 'error':
#             for remake_data in data['remake_3d']:
#                 origin_x = leds_data[origin][i]['pos'][0]
#                 origin_y = leds_data[origin][i]['pos'][1]
#                 origin_z = leds_data[origin][i]['pos'][2]
#                 target_x = leds_data[target][i]['pos'][0]
#                 target_y = leds_data[target][i]['pos'][1]
#                 target_z = leds_data[target][i]['pos'][2]
#
#                 diff_x = '%0.12f' % (origin_x - target_x)
#                 diff_y = '%0.12f' % (origin_y - target_y)
#                 diff_z = '%0.12f' % (origin_z - target_z)
#                 # print('o_t:[', diff_x, ',', diff_y, ',', diff_z, ']')
#                 distance = np.sqrt(
#                     np.power(float(diff_x), 2) + np.power(float(diff_y), 2) + np.power(
#                         float(diff_z), 2))
#                 undistort_distance_o_t.append(distance)
#                 led_num_o_t.append(f'LED {i}')
#
#                 diff_x_s = '%0.12f' % (target_x - remake_data['uncoord_s'][0][0][0])
#                 diff_y_s = '%0.12f' % (target_y - remake_data['uncoord_s'][0][0][1])
#                 diff_z_s = '%0.12f' % (target_z - remake_data['uncoord_s'][0][0][2])
#                 print('origin:[', diff_x_s, ',', diff_y_s, ',', diff_z_s, ']')
#                 s_distance = np.sqrt(
#                     np.power(float(diff_x_s), 2) + np.power(float(diff_y_s), 2) + np.power(
#                         float(diff_z_s), 2))
#                 undistort_distance_s.append(s_distance)
#
#                 diff_x_r = '%0.12f' % (target_x - remake_data['uncoord_r'][0][0][0])
#                 diff_y_r = '%0.12f' % (target_y - remake_data['uncoord_r'][0][0][1])
#                 diff_z_r = '%0.12f' % (target_z - remake_data['uncoord_r'][0][0][2])
#                 print('target:[', diff_x_r, ',', diff_y_r, ',', diff_z_r, ']')
#                 r_distance = np.sqrt(
#                     np.power(float(diff_x_r), 2) + np.power(float(diff_y_r), 2) + np.power(
#                         float(diff_z_r), 2))
#                 undistort_distance_r.append(r_distance)
#
#                 r_diff_x_s = '%0.12f' % (target_x - remake_data['r_uncoord_s'][0][0][0])
#                 r_diff_y_s = '%0.12f' % (target_y - remake_data['r_uncoord_s'][0][0][1])
#                 r_diff_z_s = '%0.12f' % (target_z - remake_data['r_uncoord_s'][0][0][2])
#                 print('simulation:[', r_diff_x_s, ',', r_diff_y_s, ',', r_diff_z_s, ']')
#                 r_distance_s = np.sqrt(
#                     np.power(float(r_diff_x_s), 2) + np.power(float(r_diff_y_s), 2) + np.power(
#                         float(r_diff_z_s), 2))
#                 r_undistort_distance_s.append(r_distance_s)
#
#                 new_x = remake_data['uncoord_r'][0][0][0]
#                 new_y = remake_data['uncoord_r'][0][0][1]
#                 new_z = remake_data['uncoord_r'][0][0][2]
#
#                 before_pts.append({'idx': i,
#                                    'pos': [new_x,
#                                            new_y,
#                                            new_z]})
#                 Pc.append([new_x,
#                            new_y,
#                            new_z])
#                 origin_pts.append(leds_data[target][i]['pos'])
#
#     plt.style.use('default')
#     plt.figure(figsize=(15, 10))
#     plt.title('Diff result')
#     markers, stemlines, baseline = plt.stem(led_num_o_t, undistort_distance_s, label='origin')
#     markers.set_color('black')
#
#     markers, stemlines, baseline = plt.stem(led_num_o_t, undistort_distance_r, label='target')
#     markers.set_color('red')
#
#     markers, stemlines, baseline = plt.stem(led_num_o_t, undistort_distance_o_t, label='oigin vs target')
#     markers.set_color('green')
#
#     markers, stemlines, baseline = plt.stem(led_num_o_t, r_undistort_distance_s, label='simulation')
#     markers.set_color('blue')
#
#     for data in Pc:
#         print(data)
#
#     A = np.vstack([np.array(Pc).T, np.ones(len(Pc))]).T
#     Rt = np.linalg.lstsq(A, np.array(origin_pts), rcond=None)[0]
#     print("Rt=\n", Rt.T)
#     # Pd_est : Pa 예측값 (Calibration 값)
#     Pd_est = np.matmul(Rt.T, A.T)
#
#     print("Pd_est=\n", Pd_est)
#     Err = np.array(origin_pts) - Pd_est.T
#     print("Err=\n", Err)
#
#     for led_num in range(LEDS_COUNT):
#         after_pts.append({'idx': led_num,
#                           'pos': Pd_est.T[led_num]})
#
#     # print('origin_pts: ', origin_pts)
#     # print('before pts: ', before_pts)
#     # print('after pts: ', after_pts)
#
#     led_num = []
#     distance_array = []
#     for i, data in enumerate(leds_data[target]):
#         if data['remake_3d'] != 'error':
#             target_x = leds_data[target][i]['pos'][0]
#             target_y = leds_data[target][i]['pos'][1]
#             target_z = leds_data[target][i]['pos'][2]
#
#             diff_x = '%0.12f' % (target_x - after_pts[i]['pos'][0])
#             diff_y = '%0.12f' % (target_y - after_pts[i]['pos'][1])
#             diff_z = '%0.12f' % (target_z - after_pts[i]['pos'][2])
#             print('lsm:[', diff_x, ',', diff_y, ',', diff_z, ']')
#             distance = np.sqrt(
#                 np.power(float(diff_x), 2) + np.power(float(diff_y), 2) + np.power(
#                     float(diff_z), 2))
#             distance_array.append(distance)
#             led_num.append(f'LED {i}')
#     # markers, stemlines, baseline = plt.stem(led_num, distance_array, label='LSM')
#     # markers.set_color('gray')
#
#     plt.legend()
#     plt.show()


def solvepnp_test_two(origin, target):
    plt.style.use('default')
    plt.rc('xtick', labelsize=5)  # x축 눈금 폰트 크기
    plt.rc('ytick', labelsize=5)
    fig_ransac = plt.figure(figsize=(15, 15))

    for cam_id, camera_info in enumerate(camera_array_target):
        fig_ransac.tight_layout()
        ax_ransac = fig_ransac.add_subplot(4, round((len(camera_array_target) + 1) / 4), cam_id + 1)
        ax_ransac.set_title(f'[{cam_id}]:', fontdict={'fontsize': 8, 'fontweight': 'medium'})
        print('cam_id ', cam_id)
        print('leds ', camera_info['model']['leds'])

        origin_leds = []
        target_leds = []
        camera_info['pts_facing']['real'] = camera_info['model']['leds']
        for led_num in camera_info['model']['leds']:
            origin_leds.append([[leds_data[origin][led_num]['pos'][0],
                                 leds_data[origin][led_num]['pos'][1],
                                 leds_data[origin][led_num]['pos'][2]]])
            target_leds.append([[leds_data[target][led_num]['pos'][0],
                                 leds_data[target][led_num]['pos'][1],
                                 leds_data[target][led_num]['pos'][2]]])

        origin_leds = np.array(origin_leds, dtype=np.float64)
        target_leds = np.array(target_leds, dtype=np.float64)

        list_2d_undistorted_origin = np.array(camera_info['undistorted_2d'], dtype=np.float64)
        # print('origin 2d undistort ', list_2d_undistorted_origin)
        list_2d_distorted_origin = np.array(camera_info['distorted_2d'], dtype=np.float64)
        # print('origin 2d distort ', list_2d_distorted_origin)

        # ox, oy = list_2d_distorted_origin.reshape(len(list_2d_distorted_origin), 2).transpose()
        # ax_ransac.scatter(ox, oy, marker='o', s=10, color='green', alpha=0.3)

        list_2d_undistorted_target = np.array(camera_array_target[cam_id]['undistorted_2d'], dtype=np.float64)
        # print('target 2d undistort ', list_2d_undistorted_target)
        list_2d_distorted_target = np.array(camera_array_target[cam_id]['distorted_2d'], dtype=np.float64)
        # print('target 2d distort ', list_2d_distorted_target)

        # Static R|T로 본 2번
        tx, ty = list_2d_distorted_target.reshape(len(list_2d_distorted_target), 2).transpose()
        ax_ransac.scatter(tx, ty, marker='o', s=10, color='red', alpha=0.5)

        # _, rvecs, tvecs = cv2.solveP3P(target_leds, list_2d_undistorted_target,
        #                                cameraK,
        #                                distCoeff,
        #                                flags=cv2.target_leds)
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(origin_leds, list_2d_undistorted_target,
                                                      cameraK,
                                                      distCoeff,
                                                      flags=cv2.SOLVEPNP_AP3P)
        ret = cv2.fisheye.projectPoints(target_leds,
                                        camera_info['S_R_T']['rvecs'],
                                        camera_info['S_R_T']['tvecs'],
                                        camera_info['cam_cal']['cameraK'],
                                        camera_info['cam_cal']['dist_coeff'])

        # 지금 보이는 화면 2번이다.
        x, y = ret[0].reshape(len(target_leds), 2).transpose()
        # ax_ransac.scatter(x, y, marker='o', s=10, color='blue', alpha=0.5)

        blobs_2d_target = []
        for i in range(len(x)):
            blobs_2d_target.append([[x[i], y[i]]])

        blobs_2d_target_distort = np.array(blobs_2d_target, dtype=np.float64)
        blobs_2d_target_undistort = cv2.fisheye.undistortPoints(blobs_2d_target_distort,
                                                                camera_info['cam_cal']['cameraK'],
                                                                camera_info['cam_cal']['dist_coeff'])

        # 좌표 numbering
        num = camera_info['model']['leds']
        for num, x, y in zip(num, x, y):
            label = '%s' % num
            ax_ransac.text(x + 2, y + 2, label, size=4, color='black')

        # pose apply 해보자
        observed_pos_r = tvecs.reshape(3)
        observed_rot_r = R.from_rotvec(rvecs.reshape(3)).as_quat()
        observed_pose_r = {'position': vector3(observed_pos_r[0], observed_pos_r[1], observed_pos_r[2]),
                           'orient': quat(observed_rot_r[0], observed_rot_r[1], observed_rot_r[2], observed_rot_r[3])}
        remake_offset_r = pose_apply_inverse(observed_pose_r, camera_array_origin[cam_id])

        print('remake offset: ', remake_offset_r)
        camera_info['remake_offset'] = remake_offset_r

        remake_leds_r = []
        for t in origin_leds:
            pt = vector3(t[0][0], t[0][1], t[0][2])
            temp_pt = transfer_point(pt, remake_offset_r)
            remake_leds_r.append([[temp_pt.x, temp_pt.y, temp_pt.z]])
        remake_leds_r = np.array(remake_leds_r, dtype=np.float64)
        tvec_r = np.array(remake_offset_r['position'])
        rot_r = R.from_quat(np.array(remake_offset_r['orient']))
        rvec_r, _ = cv2.Rodrigues(rot_r.as_matrix())

        ret = cv2.fisheye.projectPoints(remake_leds_r,
                                        camera_info['S_R_T']['rvecs'],
                                        camera_info['S_R_T']['tvecs'],
                                        camera_info['cam_cal']['cameraK'],
                                        camera_info['cam_cal']['dist_coeff'])
        xx, yy = ret[0].reshape(len(remake_leds_r), 2).transpose()

        blobs_2d_remake_r = []
        for i in range(len(xx)):
            blobs_2d_remake_r.append([[xx[i], yy[i]]])

        blobs_2d_remake_distort_r = np.array(blobs_2d_remake_r, dtype=np.float64)
        list_2d_remake_undistorted_r = cv2.fisheye.undistortPoints(blobs_2d_remake_distort_r,
                                                                   camera_info['cam_cal']['cameraK'],
                                                                   camera_info['cam_cal']['dist_coeff'])

        ax_ransac.scatter(xx, yy, marker='o', s=10, color='black', alpha=0.5)

        max_x = 0
        max_y = 0
        if abs(max(xx)) > max_x: max_x = abs(max(xx))
        if abs(min(xx)) > max_x: max_x = abs(min(xx))
        if abs(max(yy)) > max_y: max_y = abs(max(yy))
        if abs(min(yy)) > max_y: max_y = abs(min(yy))
        dimen = max(max_x, max_y)
        dimen *= 1.1
        ax_ransac.set_xlim([500, 750])
        ax_ransac.set_ylim([400, 600])



        # remake offset , transfer point 해서 투영
        camera_info['coord']['remake_undistort_s'] = list_2d_remake_undistorted_r[:, 0]
        #실제 UVC 카메라에서 찍힌 것
        camera_info['coord']['undistort_s'] = list_2d_undistorted_target

        #
        # target 좌표를 srt로 투영 시킨것
        camera_info['coord']['undistort_r'] = blobs_2d_target_undistort[:, 0]


        # # ADDDDDDDDDDDDDDDDDDDDD
        for i, coords in enumerate(blobs_2d_remake_distort_r[:, 0]):
            # print(coords)
            # print(list_2d_undistorted_target[i])

            diff_x = coords[0] - list_2d_distorted_target[i][0]
            diff_y = coords[1] - list_2d_distorted_target[i][1]

            print('diff ', diff_x, ' ', diff_y)
        #     camera_info['coord']['remake_undistort_s'][i][0] -= diff_x
        #     camera_info['coord']['remake_undistort_s'][i][1] -= diff_y

        #
        camera_info['rt']['rvecs_r'] = camera_info['S_R_T']['rvecs'][:, 0]
        camera_info['rt']['tvecs_r'] = camera_info['S_R_T']['tvecs'][:, 0]

        # camera_info['rt']['rvecs_r'] = rvecs[:, 0]
        # camera_info['rt']['tvecs_r'] = tvecs[:, 0]

    ####################
    for cam_id, data in enumerate(camera_array_target):
        # print('cam_id ', cam_id)
        # print(data)
        for i in range(len(data['pts_facing']['real']) - 1):
            facing_dot = data['pts_facing']['real'][i]
            # print(facing_dot)
            if len(data['coord']['undistort_r']) <= 0 or len(data['coord']['undistort_s']) <= 0:
                continue

            leds_data[target][facing_dot]['pair_xy'].append({'cidx': cam_id,
                                                             'led_num': facing_dot,
                                                             'scx': data['coord']['undistort_s'][i][0],
                                                             'scy': data['coord']['undistort_s'][i][1],

                                                             'rcx': data['coord']['undistort_r'][i][0],
                                                             'rcy': data['coord']['undistort_r'][i][1],

                                                             'rscx': data['coord']['remake_undistort_s'][i][0],
                                                             'rscy': data['coord']['remake_undistort_s'][i][1],

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
                undistort_s = coord_refine(camera_array_target, data[0], data[1], 'rvecs_r', 'tvecs_r', 'scx', 'scy',
                                           False)

                r_undistort_s = coord_refine(camera_array_target, data[0], data[1], 'rvecs_r', 'tvecs_r', 'rscx',
                                             'rscy',
                                             False)

                undistort_r = coord_refine(camera_array_target, data[0], data[1], 'rvecs_r', 'tvecs_r', 'rcx', 'rcy',
                                           False)

                leds_data[target][i]['remake_3d'].append(
                    {'cam_l': data[0]['cidx'], 'cam_r': data[1]['cidx'],

                     # pose offset 없음
                     'uncoord_r': undistort_r,


                     # pose offset 있음
                     'r_uncoord_s': r_undistort_s,
                     'uncoord_s': undistort_s,
                     })

    print('remake 3d result')
    led_num_o_t = []
    undistort_distance_r = []
    undistort_distance_s = []
    undistort_distance_o_t = []
    r_undistort_distance_s = []

    # for lsm
    Pc = []
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

                diff_x_s = '%0.12f' % (target_x - remake_data['uncoord_s'][0][0][0])
                diff_y_s = '%0.12f' % (target_y - remake_data['uncoord_s'][0][0][1])
                diff_z_s = '%0.12f' % (target_z - remake_data['uncoord_s'][0][0][2])
                print('origin:[', diff_x_s, ',', diff_y_s, ',', diff_z_s, ']')
                s_distance = np.sqrt(
                    np.power(float(diff_x_s), 2) + np.power(float(diff_y_s), 2) + np.power(
                        float(diff_z_s), 2))
                undistort_distance_s.append(s_distance)

                diff_x_r = '%0.12f' % (target_x - remake_data['uncoord_r'][0][0][0])
                diff_y_r = '%0.12f' % (target_y - remake_data['uncoord_r'][0][0][1])
                diff_z_r = '%0.12f' % (target_z - remake_data['uncoord_r'][0][0][2])
                # print('target:[', diff_x_r, ',', diff_y_r, ',', diff_z_r, ']')
                r_distance = np.sqrt(
                    np.power(float(diff_x_r), 2) + np.power(float(diff_y_r), 2) + np.power(
                        float(diff_z_r), 2))
                undistort_distance_r.append(r_distance)

                r_diff_x_s = '%0.12f' % (target_x - remake_data['r_uncoord_s'][0][0][0])
                r_diff_y_s = '%0.12f' % (target_y - remake_data['r_uncoord_s'][0][0][1])
                r_diff_z_s = '%0.12f' % (target_z - remake_data['r_uncoord_s'][0][0][2])
                # print('simulation:[', r_diff_x_s, ',', r_diff_y_s, ',', r_diff_z_s, ']')
                r_distance_s = np.sqrt(
                    np.power(float(r_diff_x_s), 2) + np.power(float(r_diff_y_s), 2) + np.power(
                        float(r_diff_z_s), 2))
                r_undistort_distance_s.append(r_distance_s)

                new_x = remake_data['r_uncoord_s'][0][0][0]
                new_y = remake_data['r_uncoord_s'][0][0][1]
                new_z = remake_data['r_uncoord_s'][0][0][2]

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
    markers, stemlines, baseline = plt.stem(led_num_o_t, undistort_distance_s, label='UVC')
    markers.set_color('black')

    markers, stemlines, baseline = plt.stem(led_num_o_t, undistort_distance_r, label='target')
    markers.set_color('red')

    markers, stemlines, baseline = plt.stem(led_num_o_t, undistort_distance_o_t, label='oigin vs target')
    markers.set_color('green')

    # markers, stemlines, baseline = plt.stem(led_num_o_t, r_undistort_distance_s, label='remake')
    # markers.set_color('blue')

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
    # markers, stemlines, baseline = plt.stem(led_num, distance_array, label='LSM')
    # markers.set_color('gray')




def diff_check(origin, target):
    print('check_diff')
    for cam_id, camera_info in enumerate(camera_array_target):
        origin_leds = []
        target_leds = []
        camera_info['pts_facing']['real'] = camera_info['model']['leds']
        offset = camera_info['remake_offset']
        for i, led_num in enumerate(camera_info['model']['leds']):
            if i == 3:
                continue

            origin_leds.append([[leds_data[origin][led_num]['pos'][0],
                                 leds_data[origin][led_num]['pos'][1],
                                 leds_data[origin][led_num]['pos'][2]]])
            target_leds.append([[leds_data[target][led_num]['pos'][0],
                                 leds_data[target][led_num]['pos'][1],
                                 leds_data[target][led_num]['pos'][2]]])
            diff_x = '%0.12f' % (leds_data[origin][led_num]['pos'][0] - leds_data[target][led_num]['pos'][0])
            diff_y = '%0.12f' % (leds_data[origin][led_num]['pos'][1] - leds_data[target][led_num]['pos'][1])
            diff_z = '%0.12f' % (leds_data[origin][led_num]['pos'][2] - leds_data[target][led_num]['pos'][2])
            for remake_data in leds_data[target][led_num]['remake_3d']:
                # print(remake_data)
                p_o = vector3(leds_data[origin][led_num]['pos'][0], leds_data[origin][led_num]['pos'][1], leds_data[origin][led_num]['pos'][2])
                p_n = vector3(remake_data['r_uncoord_s'][0][0][0],
                              remake_data['r_uncoord_s'][0][0][1],
                              remake_data['r_uncoord_s'][0][0][2])
                tmp_p_o = transfer_point_inverse(p_o, offset)
                tmp_p_n = transfer_point_inverse(p_n, offset)
                # x = tmp_p_o.x - tmp_p_n.x
                # y = tmp_p_o.y - tmp_p_n.y
                # z = tmp_p_o.z - tmp_p_n.z
                x = tmp_p_n.x - p_o.x
                y = tmp_p_n.y - p_o.y
                z = tmp_p_n.z - p_o.z
                print('cam_id ', cam_id, ' led_num ', led_num)
                print('origin ', leds_data[origin][led_num]['pos'])
                print('target ', leds_data[target][led_num]['pos'])
                # rx = remake_data['r_uncoord_s'][0][0][0] + x
                # ry = remake_data['r_uncoord_s'][0][0][1] + y
                # rz = remake_data['r_uncoord_s'][0][0][2] + z
                rx = leds_data[origin][led_num]['pos'][0] + x
                ry = leds_data[origin][led_num]['pos'][1] + y
                rz = leds_data[origin][led_num]['pos'][2] + z

                diff_x_1 = '%0.12f' % (leds_data[target][led_num]['pos'][0] - rx)
                diff_y_1 = '%0.12f' % (leds_data[target][led_num]['pos'][1] - ry)
                diff_z_1 = '%0.12f' % (leds_data[target][led_num]['pos'][2] - rz)
                print('o-t: ', ' x', diff_x, ' y', diff_y, ' z', diff_z)
                # print('diff: ', ' x', x, ' y', y, ' z', z)
                # print('final: ', ' rx', diff_x_1, ' ry', diff_y_1, ' rz', diff_z_1)

        # origin_leds = np.array(origin_leds, dtype=np.float64)
        # target_leds = np.array(target_leds, dtype=np.float64)


#
#
# def cal_diff_data():
#     length = len(leds_dic['pts'])
#     pts_final = []
#
#     for i in range(length):
#
#         p_o = vector3(leds_dic['pts_origin_offset'][i]['pos'][0], leds_dic['pts_origin_offset'][i]['pos'][1], leds_dic['pts_origin_offset'][i]['pos'][2])
#         p_n = vector3(leds_dic['pts_noise_refactor'][i]['hcoord'][0], leds_dic['pts_noise_refactor'][i]['hcoord'][1], leds_dic['pts_noise_refactor'][i]['hcoord'][2])
#
#         tmp_p_o = transfer_point_inverse(p_o, offset)
#         tmp_p_n = transfer_point_inverse(p_n, offset)
#
#         x = tmp_p_o.x - tmp_p_n.x
#         y = tmp_p_o.y - tmp_p_n.y
#         z = tmp_p_o.z - tmp_p_n.z
#
#         '''
#         x = leds_dic['pts_origin_offset'][i]['pos'][0] - leds_dic['pts_noise_refactor'][i]['hcoord'][0]
#         y = leds_dic['pts_origin_offset'][i]['pos'][1] - leds_dic['pts_noise_refactor'][i]['hcoord'][1]
#         z = leds_dic['pts_origin_offset'][i]['pos'][2] - leds_dic['pts_noise_refactor'][i]['hcoord'][2]
#         '''
#
#         rx = leds_dic['pts_noise'][i]['pos'][0] + x
#         ry = leds_dic['pts_noise'][i]['pos'][1] + y
#         rz = leds_dic['pts_noise'][i]['pos'][2] + z
#
#         u = leds_dic['pts_noise'][i]['dir'][0]
#         v = leds_dic['pts_noise'][i]['dir'][1]
#         w = leds_dic['pts_noise'][i]['dir'][2]
#
#         diff_x = leds_dic['pts'][i]['pos'][0] - leds_dic['pts_noise'][i]['pos'][0]
#         diff_y = leds_dic['pts'][i]['pos'][1] - leds_dic['pts_noise'][i]['pos'][1]
#         diff_z = leds_dic['pts'][i]['pos'][2] - leds_dic['pts_noise'][i]['pos'][2]
#
#         if debug == ENABLE:
#             print('1.[', i, ']: x(', diff_x, ') y(', diff_y, ') z(', diff_z, ')')
#             print('2.[', i, ']: x(', x, ') y(', y, ') z(', z, ')')
#
#         pts_final.append({'idx': i, 'pos': list(map(float, [rx, ry, rz])),
#                           'dir': list(map(float, [u, v, w])),
#                           'pattern': leds_dic['pts'][i]['pattern']})
#
#         leds_dic['pts_final'] = pts_final

def init_coord_json(path, file):
    print('start ', init_coord_json.__name__)
    try:
        json_file = open(''.join([path, f'{file}']))
        jsonObject = json.load(json_file)
        model_points = jsonObject.get('TrackedObject').get('ModelPoints')
        pts = [0 for i in range(len(model_points))]
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
            pts[int(idx)] = {'idx': idx,
                             'pos': [x, y, z],
                             'dir': [u, v, w],
                             'res': [r1, r2, r3],
                             'pair_xy': [],
                             'remake_3d': []}

            print(''.join(['{ .pos = {{', f'{x}', ',', f'{y}', ',', f'{z}',
                           ' }}, .dir={{', f'{u}', ',', f'{v}', ',', f'{w}', ' }}, .pattern=', f'{idx}', '},']))
    except:
        print('exception')
        traceback.print_exc()
    else:
        print('done')
    finally:
        print('done')
    return pts

def color_loop(idx):
    if idx % 2 == 0:
        return 'red'
    else:
        return 'blue'

def facing_dot_area(cam_array, origin):
    # plt.style.use('default')
    # fig_ransac = plt.figure(figsize=(15, 15))
    cam_id_array = []
    rer_array = []
    drer_array = []

    dist_to_controller = 0.5

    VIRTUAL = 1
    cam_pose = []
    if VIRTUAL == 1:
        horizon_min_degree = 10
        vertical_min_degree = 30
        hcnt = round(180 / horizon_min_degree)
        vcnt = round(360 / vertical_min_degree)
        cam_id = 0;
        for v in range(vcnt):
            for h in range(hcnt):
                cam_pose.append({
                    'cidx': cam_id,
                    'position': vector3(0.0, 0.0, dist_to_controller),
                    'orient': get_quat_from_euler('xyz', [h * horizon_min_degree, 0, v * vertical_min_degree])
                })
                cam_id += 1
                print(cam_pose[h]['orient'])
    else:
        for idx, cam_array_data in enumerate(cam_array):
            cam_pose.append({
                'cidx': cam_array_data['cidx'],
                'position': cam_array_data['position'],
                'orient': cam_array_data['orient']
            })


    # draw camera position
    for i, cam_pose_data in enumerate(cam_pose):
        new_cam_pose = {'cidx': cam_pose_data['cidx'], 'position': cam_pose_data['position'],
                        'orient': cam_pose_data['orient']}
        new_pt = camtoworld(new_cam_pose['position'], new_cam_pose)
        ax.scatter(new_pt.x, new_pt.y, new_pt.z, marker='*', color='blue', s=20)
        ax.text(new_pt.x, new_pt.y, new_pt.z, cam_pose_data['cidx'], size=10)

        print('cam_id ', cam_pose_data['cidx'],
              ' pos: ', cam_pose_data['position'],
              ' ori(euler): ', np.round_(get_euler_from_quat('xyz', cam_pose_data['orient']), 3))

    cam = 0
    for cam_num, cam_pose_data in enumerate(cam_pose):
        length = round((len(cam_pose) + 1) / 4)
        if length == 0:
            length = 1
        # ax_ransac = fig_ransac.add_subplot(4, length, cam_num + 1)
        o_pts_facing, leds_array = np.array(check_facing_dot(leds_data[origin], cam_pose_data))
        cnt = 0

        # print('f: ', leds_array)
        # print('cam id: ', cam_pose_data['cidx'], ' ori: ', cam_pose_data)
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
        if VIRTUAL == 1:
            cam_num = cam
        oret = cv2.projectPoints(origin_leds, Rod, obj_cam_pos_n,
                                 cam_array[cam_num]['cam_cal']['cameraK'],
                                 cam_array[cam_num]['cam_cal']['dist_coeff'])
        o_xx, o_yy = oret[0].reshape(len(origin_leds), 2).transpose()
        blobs_2d_origin = []
        for i in range(len(o_xx)):
            blobs_2d_origin.append([[o_xx[i], o_yy[i]]])

        print(blobs_2d_origin)
        #
        # origin_pts = []
        # for i in range(len(o_xx)):
        #     _opos = list(map(float, [o_xx[i], o_yy[i]]))
        #     origin_pts.append({'idx': leds_array[i], 'pos': _opos, 'reserved': 0})
        #
        # color = color_loop(cam_num)
        # draw_dots(2, origin_pts, ax_ransac, color)
        #
        # max_x = 0
        # max_y = 0
        # if abs(max(o_xx)) > max_x: max_x = abs(max(o_xx))
        # if abs(min(o_xx)) > max_x: max_x = abs(min(o_xx))
        # if abs(max(o_yy)) > max_y: max_y = abs(max(o_yy))
        # if abs(min(o_yy)) > max_y: max_y = abs(min(o_yy))
        # dimen = max(max_x, max_y)
        # dimen *= 1.1
        # ax_ransac.set_xlim([0, 1280])
        # ax_ransac.set_ylim([0, 960])
        #
        #
        # blobs_2d = np.array(blobs_2d_origin, dtype=np.float64)
        # blobs_2d = blobs_2d[:, 0]






if __name__ == "__main__":
    print('start ransac test main')
    plt.style.use('default')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    origin = 'rift_6'
    origin_r = 'rift_6_right'
    target = 'rift_2'
    # read origin data
    read_led_pts(origin)
    # read target data
    read_led_pts(target)
    leds_data[origin_r] = init_coord_json('../jsons/', 'rifts6_right.json')

    draw_dots(3, leds_data[origin], ax, 'blue')
    # draw_dots(3, leds_data[target], ax, 'red')
    # draw_dots(3, leds_data[origin_r], ax, 'red')
    draw_blobs(ax)
    print('origin')
    for i, led in enumerate(leds_data[origin]):
        print('led_num ', int(led['idx']))
        origin_dis = np.sqrt(np.power(led['pos'][0], 2) + np.power(led['pos'][1], 2) +
                             np.power(led['pos'][2], 2))
        target_dis = np.sqrt(np.power(leds_data[target][i]['pos'][0], 2) + np.power(leds_data[target][i]['pos'][1], 2) +
                             np.power(leds_data[target][i]['pos'][2], 2))
        print('dis: ', origin_dis, 'vs', target_dis)
        print('diff: ', '%0.8f' % np.abs(origin_dis - target_dis))
        # print(led)
    # print('target')
    # for led in leds_data[target]:
    # print(led)

    path_1 = "../../new_cal_project/cal_project/cal_uvc/jsons/"
    path_2 = '../backup/jsons/backup/good_6_2/'
    path_3 = '../backup/jsons/backup/static_6/'
    path_4 = '../backup/jsons/backup/static_6_2/'
    camera_array_origin = init_model_json(path_3)
    camera_array_target = init_model_json(path_4)

    facing_dot_area(camera_array_origin, origin)

    # solvepnp_test(origin, target)
    # solvepnp_test_two(origin, target)
    # diff_check(origin, target)









    plt.legend()
    plt.show()
