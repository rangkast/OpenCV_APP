import matplotlib.pyplot as plt
import numpy as np

from definition import *
import itertools
import math
import pandas as pd
import sys
import cv2
from uvc_openCV import *
from collections import OrderedDict


def print_result(title, target):
    leds_dic['after_led'] = [{'idx': -1} for i in range(LED_COUNT)]
    leds_dic['before_led'] = [{'idx': -1} for i in range(LED_COUNT)]

    lsm_and_print_result(title, target, target)

    print('before led')
    before_array = []
    for idx, info in enumerate(leds_dic['before_led']):
        if info['idx'] == -1:
            continue
        i = int(info['idx'])
        x = round(leds_dic['before_led'][i]['pos'][0], 8)
        y = round(leds_dic['before_led'][i]['pos'][1], 8)
        z = round(leds_dic['before_led'][i]['pos'][2], 8)
        u = leds_dic['target_pts'][i]['dir'][0]
        v = leds_dic['target_pts'][i]['dir'][1]
        w = leds_dic['target_pts'][i]['dir'][2]
        before_array.append({i: [x, y, z, u, v, w]})
        print(''.join(['{ .pos = {{', f'{x}', ',', f'{y}', ',', f'{z}',
                       ' }}, .dir={{', f'{u}', ',', f'{v}', ',', f'{w}', ' }}, .pattern=', f'{i}', '}, ']))

    print('after led')
    after_array = []
    x_temp_array = []
    y_temp_array = []
    z_temp_array = []
    eu_temp_array = []
    for idx, info in enumerate(leds_dic['after_led']):
        if info['idx'] == -1:
            continue
        i = int(info['idx'])
        x = round(leds_dic['after_led'][i]['pos'][0], 8)
        y = round(leds_dic['after_led'][i]['pos'][1], 8)
        z = round(leds_dic['after_led'][i]['pos'][2], 8)
        u = leds_dic['target_pts'][i]['dir'][0]
        v = leds_dic['target_pts'][i]['dir'][1]
        w = leds_dic['target_pts'][i]['dir'][2]
        after_array.append({i: [x, y, z, u, v, w]})
        print(''.join(['{ .pos = {{', f'{x}', ',', f'{y}', ',', f'{z}',
                       ' }}, .dir={{', f'{u}', ',', f'{v}', ',', f'{w}', ' }}, .pattern=', f'{i}', '}, ']))

        ox = leds_dic['target_pts_4'][i]['pos'][0]
        oy = leds_dic['target_pts_4'][i]['pos'][1]
        oz = leds_dic['target_pts_4'][i]['pos'][2]

        x_temp_array.append(np.abs(x - ox))
        y_temp_array.append(np.abs(y - oy))
        z_temp_array.append(np.abs(z - oz))
        eu_temp_array.append(
            distance.euclidean(np.array(leds_dic['after_led'][i]['pos']), np.array(leds_dic['target_pts_4'][i]['pos'])))

    print("X_diff_val")
    print(x_temp_array)
    print("MAX LED Index:", np.argmax(np.array(x_temp_array)), "Distance:", max(x_temp_array))
    print("Y_diff_val")
    print(y_temp_array)
    print("MAX LED Index:", np.argmax(np.array(y_temp_array)), "Distance:", max(y_temp_array))
    print("Z_diff_val")
    print(z_temp_array)
    print("MAX LED Index:", np.argmax(np.array(z_temp_array)), "Distance:", max(z_temp_array))
    print("euclidean_diff_val")
    print(eu_temp_array)
    print("MAX LED Index:", np.argmax(np.array(eu_temp_array)), "Distance:", max(eu_temp_array))

    target_json_file = ''.join(['jsons/specs/', f'{TARGET}'])
    cal_json_file = ''.join(['cal/', f'{(TARGET.split("."))[0]}'])
    cal_json_file += "+"
    change_json_with(target_json_file, cal_json_file, after_array)

    json_file = ''.join(['jsons/test_result/', f'{title}'])
    group_data = OrderedDict()
    group_data['before'] = before_array
    group_data['after'] = after_array
    rw_json_data(WRITE, json_file, group_data)


def grouping_remake_3d(target):
    group_remake_array = [0 for i in range(LED_COUNT)]
    for cam_data in leds_dic['cam_info']:
        group_num = int(cam_data['group'])
        for i in range(len(leds_dic['cam_info'][group_num]['model']['spec']) - 1):
            # print(leds_dic['cam_info'][cam_id]['model']['spec'][i]['idx'])
            led_num = int(leds_dic['cam_info'][group_num]['model']['spec'][i]['idx'])
            for remake_data in leds_dic[target][led_num]['remake_3d']:
                if remake_data != 'error':
                    if remake_data['cam_r'] == remake_data['cam_l'] + 1 and \
                            remake_data['cam_r'] == group_num + 1:
                        print('detect ', led_num)
                        print(group_num)
                        print(remake_data)
                        group_remake_array[led_num] = remake_data

    for i, group_remake in enumerate(group_remake_array):
        leds_dic[target][i]['remake_3d'].clear()
        leds_dic[target][i]['remake_3d'].append(group_remake)


def best_ecc_remake_3d(target, set_status, led_num):
    temp_pair_array = leds_dic[target][led_num]['pair_xy']
    sorted_ecc = sorted(temp_pair_array, key=lambda eccentricity: (eccentricity['ecc']))

    result = coordRefactor(leds_dic['cam_info'], sorted_ecc[0], sorted_ecc[1], set_status)
    leds_dic[target][led_num]['remake_3d'].clear()
    leds_dic[target][led_num]['remake_3d'].append(
        {'cam_l': sorted_ecc[0]['cidx'], 'cam_r': sorted_ecc[1]['cidx'], 'coord': result})
    print("=== best_ecc_remake_3d ====")
    print("pair_array")
    print(temp_pair_array)
    print("sorted_ecc")
    print(sorted_ecc)
    print("led num:", led_num, "result:", result)


def inlier_loop_static(blobs, target, json_data):
    # refactoring 3d points
    compare_leds = []
    for i in range(len(leds_dic[target])):
        led_pair_cnt = len(leds_dic[target][i]['pair_xy'])
        if led_pair_cnt < 2:
            print(f'Error LED Num {i} has no more than 2-cameras')
            leds_dic[target][i]['remake_3d'] = 'error'
        else:
            comb_led = list(itertools.combinations(leds_dic[target][i]['pair_xy'], 2))
            for data in comb_led:
                compare_leds.append(i)

    compare_leds_result = []
    for value in compare_leds:
        if value not in compare_leds_result:
            compare_leds_result.append(value)
    # load json file
    json_file = ''.join(['jsons/inlier_loop/', f'{json_data}'])
    jdata = rw_json_data(READ, json_file, None)

    for led_num in compare_leds_result:
        print('try to num: ', led_num)
        lcam = jdata[f'{led_num}']['remake']['cam_l']
        lrvecs = jdata[f'{led_num}']['remake']['lrvecs']
        ltvecs = jdata[f'{led_num}']['remake']['ltvecs']
        lcx = -1
        lcy = -1
        for i, data in enumerate(leds_dic['cam_info'][lcam][blobs]):
            if led_num == data['idx']:
                if DO_UNDISTORT == ENABLE:
                    lcx = leds_dic['cam_info'][lcam]['undistorted_2d'][i][0][0]
                    lcy = leds_dic['cam_info'][lcam]['undistorted_2d'][i][0][1]
                else:
                    lcx = data['cx']
                    lcy = data['cy']
                break

        rcam = jdata[f'{led_num}']['remake']['cam_r']
        rrvecs = jdata[f'{led_num}']['remake']['rrvecs']
        rtvecs = jdata[f'{led_num}']['remake']['rtvecs']
        rcx = -1
        rcy = -1
        for i, data in enumerate(leds_dic['cam_info'][rcam][blobs]):
            if led_num == data['idx']:
                if DO_UNDISTORT == ENABLE:
                    rcx = leds_dic['cam_info'][rcam]['undistorted_2d'][i][0][0]
                    rcy = leds_dic['cam_info'][rcam]['undistorted_2d'][i][0][1]
                else:
                    rcx = data['cx']
                    rcy = data['cy']
                break

        print('lcam : ', lcam, ' ', lcx, ',', lcy, ' r:', lrvecs, ' t:', ltvecs)
        print('rcam : ', rcam, ' ', rcx, ',', rcy, ' r:', rrvecs, ' t:', rtvecs)
        LRVECS = np.array([[lrvecs[0]], [lrvecs[1]], [lrvecs[2]]], dtype=np.float64)
        LTVECS = np.array([[ltvecs[0]], [ltvecs[1]], [ltvecs[2]]], dtype=np.float64)

        RRVECS = np.array([[rrvecs[0]], [rrvecs[1]], [rrvecs[2]]], dtype=np.float64)
        RTVECS = np.array([[rtvecs[0]], [rtvecs[1]], [rtvecs[2]]], dtype=np.float64)

        left_rotation, jacobian = cv2.Rodrigues(LRVECS)
        right_rotation, jacobian = cv2.Rodrigues(RRVECS)

        # projection matrices:
        RT = np.zeros((3, 4))
        RT[:3, :3] = left_rotation
        RT[:3, 3] = LTVECS.transpose()
        if DO_UNDISTORT == ENABLE:
            left_projection = np.dot(cameraK, RT)
        else:
            left_projection = np.dot(leds_dic['cam_info'][lcam]['cam_cal']['cameraK'], RT)

        RT = np.zeros((3, 4))
        RT[:3, :3] = right_rotation
        RT[:3, 3] = RTVECS.transpose()
        if DO_UNDISTORT == ENABLE:
            right_projection = np.dot(cameraK, RT)
        else:
            right_projection = np.dot(leds_dic['cam_info'][rcam]['cam_cal']['cameraK'], RT)

        triangulation = cv2.triangulatePoints(left_projection, right_projection,
                                              (lcx, lcy),
                                              (rcx, rcy))
        homog_points = triangulation.transpose()

        get_points = cv2.convertPointsFromHomogeneous(homog_points)

        leds_dic[target][led_num]['remake_3d'].append({'cam_l': lcam, 'cam_r': rcam, 'coord': get_points})

    # for title, data in jdata["12"]['remake'].items():
    #     print("  * %s: %s" % (title, data))
    #


def inlier_loop(target, json_data):
    # print datas
    for cam_id, leds in enumerate(leds_dic['cam_info']):
        print('\n\n')
        print('cam_id: ', cam_id)
        candidatas_cnt = 0
        for blob_info in leds['inliers_loop']:
            print(blob_info['inliers'][:, 0], ' rer: ', blob_info['rer'])
            for candidate_idx in range(len(blob_info['candidates']) - 1):
                candidatas = blob_info['candidates'][candidate_idx]
                # print(candidatas)

                # 일단은 led 좌표가 같다. 추후에 variation을 확인해 보자
                if candidatas_cnt == 0:
                    led_num = candidatas['idx']
                    distorted_x = candidatas['cx']
                    distorted_y = candidatas['cy']
                    leds_dic[target][led_num]['pair_xy'].append({'cidx': cam_id,
                                                                 'led_num': led_num,
                                                                 'cx': distorted_x,
                                                                 'cy': distorted_y})
            candidatas_cnt += 1
            print('rvecs: ', blob_info['rvecs'][:, 0])
            print('tvecs: ', blob_info['tvecs'][:, 0])
    # refactoring 3d points
    compare_leds = []
    for i in range(len(leds_dic[target])):
        led_pair_cnt = len(leds_dic[target][i]['pair_xy'])
        if led_pair_cnt < 2:
            print(f'Error LED Num {i} has no more than 2-cameras')
            leds_dic[target][i]['remake_3d'] = 'error'
        else:
            comb_led = list(itertools.combinations(leds_dic[target][i]['pair_xy'], 2))
            for data in comb_led:
                l_cam = leds_dic['cam_info'][data[0]['cidx']]
                r_cam = leds_dic['cam_info'][data[1]['cidx']]

                l_group = leds_dic['cam_info'][data[0]['cidx']]['group']
                r_group = leds_dic['cam_info'][data[1]['cidx']]['group']
                if l_group != r_group:
                    continue
                # if DEBUG == ENABLE:
                #     print('comb_led idx: ', i, ' : ', data)
                compare_leds.append(i)

                # pose가 4X4개가 있음
                l_rt_candidate = []
                r_rt_candidate = []
                for left_blob_info in l_cam['inliers_loop']:
                    l_rt_candidate.append([left_blob_info['rvecs'][:, 0].tolist(),
                                           left_blob_info['tvecs'][:, 0].tolist()])
                for right_blob_info in r_cam['inliers_loop']:
                    r_rt_candidate.append([right_blob_info['rvecs'][:, 0].tolist(),
                                           right_blob_info['tvecs'][:, 0].tolist()])

                for left_blob_info in l_cam['inliers_loop']:
                    left_rotation, jacobian = cv2.Rodrigues(left_blob_info['rvecs'])
                    for right_blob_info in r_cam['inliers_loop']:
                        right_rotation, jacobian = cv2.Rodrigues(right_blob_info['rvecs'])

                        # projection matrices:
                        RT = np.zeros((3, 4))
                        RT[:3, :3] = left_rotation
                        RT[:3, 3] = left_blob_info['tvecs'].transpose()
                        if DO_UNDISTORT == ENABLE:
                            left_projection = np.dot(cameraK, RT)
                        else:
                            left_projection = np.dot(l_cam['cam_cal']['cameraK'], RT)

                        RT = np.zeros((3, 4))
                        RT[:3, :3] = right_rotation
                        RT[:3, 3] = right_blob_info['tvecs'].transpose()
                        if DO_UNDISTORT == ENABLE:
                            right_projection = np.dot(cameraK, RT)
                        else:
                            right_projection = np.dot(r_cam['cam_cal']['cameraK'], RT)

                        triangulation = cv2.triangulatePoints(left_projection, right_projection,
                                                              (data[0]['cx'], data[0]['cy']),
                                                              (data[1]['cx'], data[1]['cy']))
                        homog_points = triangulation.transpose()

                        get_points = cv2.convertPointsFromHomogeneous(homog_points)

                        leds_dic[target][i]['remake_3d'].append(
                            {'cam_l': data[0]['cidx'],
                             'lrvecs': left_blob_info['rvecs'][:, 0].tolist(),
                             'ltvecs': left_blob_info['tvecs'][:, 0].tolist(),
                             'lcx': data[0]['cx'], 'lcy': data[0]['cy'],
                             'linlier': left_blob_info['inliers'][:, 0].tolist(),
                             'lrt_candidate': l_rt_candidate,
                             'lrer': left_blob_info['rer'],

                             'cam_r': data[1]['cidx'],
                             'rrvecs': right_blob_info['rvecs'][:, 0].tolist(),
                             'rtvecs': right_blob_info['tvecs'][:, 0].tolist(),
                             'rcx': data[1]['cx'], 'rcy': data[1]['cy'],
                             'rinlier': right_blob_info['inliers'][:, 0].tolist(),
                             'rrt_candidate': r_rt_candidate,
                             'rrer': right_blob_info['rer'],

                             'coord': get_points.tolist()})
                # for remake in leds_dic[target][i]['remake_3d']:
                #     print(remake)
    # draw result

    plt.style.use('default')
    plt.figure(figsize=(20, 20))
    plt.title('inlier loop test')
    led_distance = []
    led_num = []

    for idx, i in enumerate(compare_leds):
        # print('led num: ', i)
        # print('origin: ', leds_dic[target][i]['pos'])

        for re_idx, remake in enumerate(leds_dic[target][i]['remake_3d']):
            # print('remake: ', remake)
            diff_x = '%0.12f' % (
                    leds_dic[target][i]['pos'][0] - remake['coord'][0][0][0])
            diff_y = '%0.12f' % (
                    leds_dic[target][i]['pos'][1] - remake['coord'][0][0][1])
            diff_z = '%0.12f' % (
                    leds_dic[target][i]['pos'][2] - remake['coord'][0][0][2])

            distance = '%0.12f' % np.sqrt(
                np.power(float(diff_x), 2) + np.power(float(diff_y), 2) + np.power(
                    float(diff_z), 2))
            # print('D:[', diff_x, ',', diff_y, ',', diff_z, ']', ' ', distance)

            if float(distance) < leds_dic[target][i]['loop_pos']['dis']:
                leds_dic[target][i]['loop_pos'] = {'rx': remake['coord'][0][0][0],
                                                   'ry': remake['coord'][0][0][1],
                                                   'rz': remake['coord'][0][0][2],
                                                   'dx': diff_x, 'dy': diff_y, 'dz': diff_z,
                                                   'dis': float(distance),
                                                   'remake': remake}

            # target 0.1mm
            if float(distance) < 0.0001:
                led_distance.append(float(distance))
                led_num.append(f'LED {i}')

    markers, stemlines, baseline = plt.stem(led_num, led_distance, use_line_collection=True)
    markers.set_color('red')

    plt.figure(figsize=(20, 20))
    plt.title('result')

    compare_leds_result = []
    led_num = []
    led_distance = []
    for value in compare_leds:
        if value not in compare_leds_result:
            compare_leds_result.append(value)

    for i in compare_leds_result:
        data = leds_dic[target][i]
        print('led_num: ', i)
        print('o: ', data['pos'][0], ',', data['pos'][1], ',', data['pos'][2])
        print('r: ', data['loop_pos']['rx'], ',', data['loop_pos']['ry'], ',', data['loop_pos']['rz'])
        print('d: ', data['loop_pos']['dx'], ',', data['loop_pos']['dy'], ',', data['loop_pos']['dz'])
        print('dis: ', '%0.8f' % data['loop_pos']['dis'])
        print('left cam info ', data['loop_pos']['remake']['cam_l'])
        print('lrvecs: ', data['loop_pos']['remake']['lrvecs'], ' ltvecs: ',
              data['loop_pos']['remake']['ltvecs'])
        print('linliers: ', data['loop_pos']['remake']['linlier'], ' lrer: ', data['loop_pos']['remake']['lrer'])
        print('l_rt_candidate: ', data['loop_pos']['remake']['lrt_candidate'])

        print('right cam info ', data['loop_pos']['remake']['cam_r'])
        print('rrvecs: ', data['loop_pos']['remake']['rrvecs'], ' rtvecs: ',
              data['loop_pos']['remake']['rtvecs'])
        print('rinliers: ', data['loop_pos']['remake']['rinlier'], ' rrer: ', data['loop_pos']['remake']['rrer'])
        print('r_rt_candidate: ', data['loop_pos']['remake']['rrt_candidate'])
        print('\n')
        led_distance.append(data['loop_pos']['dis'])
        led_num.append(f'LED {i}')
        data['remake_3d'].append(data['loop_pos']['remake'])

    for i in range(LED_COUNT):
        diff_x = '%0.12f' % (
                leds_dic['pts'][i]['pos'][0] - leds_dic['target_pts'][i]['pos'][0])
        diff_y = '%0.12f' % (
                leds_dic['pts'][i]['pos'][1] - leds_dic['target_pts'][i]['pos'][1])
        diff_z = '%0.12f' % (
                leds_dic['pts'][i]['pos'][2] - leds_dic['target_pts'][i]['pos'][2])
        distance = '%0.8f' % np.sqrt(
            np.power(float(diff_x), 2) + np.power(float(diff_y), 2) + np.power(
                float(diff_z), 2))
        print('p: ', leds_dic['pts'][i]['pos'][0], ',', leds_dic['pts'][i]['pos'][1], ',', leds_dic['pts'][i]['pos'][2])
        print('s: ', leds_dic['target_pts'][i]['pos'][0], ',', leds_dic['target_pts'][i]['pos'][1], ',',
              leds_dic['target_pts'][i]['pos'][2])
        print(i, '  D:[', diff_x, ',', diff_y, ',', diff_z, ']', ' ', distance)

    # Ready for data
    group_data = OrderedDict()
    for i in compare_leds_result:
        print(leds_dic[target][i]['loop_pos'])
        group_data[f'{i}'] = leds_dic[target][i]['loop_pos']

    # Print JSON
    # print(json.dumps(group_data, ensure_ascii=False, indent="\t"))

    json_file = ''.join(['jsons/inlier_loop/', f'{json_data}'])
    rw_json_data(WRITE, json_file, group_data)

    markers, stemlines, baseline = plt.stem(led_num, led_distance, use_line_collection=True)
    markers.set_color('red')


def pairPoints(key, target):
    # print('start ', pairPoints.__name__)
    ecc = -1
    for idx, leds in enumerate(leds_dic['cam_info']):
        cam_id = idx
        for i in range(len(leds[key]) - 1):
            led_num = int(leds[key][i]['idx'])
            if DO_UNDISTORT == ENABLE:
                cx = leds['undistorted_2d'][i][0][0]
                cy = leds['undistorted_2d'][i][0][1]
            else:
                cx = leds[key][i]['cx']
                cy = leds[key][i]['cy']

            if DEBUG == ENABLE:
                print('cam: ', cam_id, ' led_num: ', led_num, ' ', leds[key][i])

            leds_dic[target][led_num]['pair_xy'].append({'cidx': cam_id,
                                                         'led_num': led_num,
                                                         'cx': cx,
                                                         'cy': cy})

    if DEBUG == ENABLE:
        for i in range(len(leds_dic[target])):
            print(leds_dic[target][i]['pair_xy'])


def projection_matrix(l_cam, r_cam, rt_status):
    key = ret_key(rt_status)
    left_rotation, jacobian = cv2.Rodrigues(l_cam[key]['rvecs'])
    right_rotation, jacobian = cv2.Rodrigues(r_cam[key]['rvecs'])

    # projection matrices:
    RT = np.zeros((3, 4))
    RT[:3, :3] = left_rotation
    RT[:3, 3] = l_cam[key]['tvecs'].transpose()
    if DO_UNDISTORT == ENABLE:
        left_projection = np.dot(cameraK, RT)
    else:
        left_projection = np.dot(l_cam['cam_cal']['cameraK'], RT)

    RT = np.zeros((3, 4))
    RT[:3, :3] = right_rotation
    RT[:3, 3] = r_cam[key]['tvecs'].transpose()
    if DO_UNDISTORT == ENABLE:
        right_projection = np.dot(cameraK, RT)
    else:
        right_projection = np.dot(r_cam['cam_cal']['cameraK'], RT)

    return left_projection, right_projection


def coordRefactor(cam_info, camera_l, camera_r, rt_status):
    left_projection, right_projection = projection_matrix(cam_info[camera_l['cidx']],
                                                          cam_info[camera_r['cidx']],
                                                          rt_status)

    triangulation = cv2.triangulatePoints(left_projection, right_projection,
                                          (camera_l['cx'], camera_l['cy']),
                                          (camera_r['cx'], camera_r['cy']))
    homog_points = triangulation.transpose()
    get_points = cv2.convertPointsFromHomogeneous(homog_points)

    return get_points


def coord2dto3d(target, set_status):
    for i in range(len(leds_dic[target])):
        led_pair_cnt = len(leds_dic[target][i]['pair_xy'])
        if led_pair_cnt < 2:
            print(f'Error LED Num {i} has no more than 2-cameras')
            leds_dic[target][i]['remake_3d'] = 'error'
        else:
            comb_led = list(itertools.combinations(leds_dic[target][i]['pair_xy'], 2))
            for data in comb_led:
                if DEBUG == ENABLE:
                    print('comb_led idx: ', i, ' : ', data)
                result = coordRefactor(leds_dic['cam_info'], data[0], data[1], set_status)
                leds_dic[target][i]['remake_3d'].append(
                    {'cam_l': data[0]['cidx'], 'cam_r': data[1]['cidx'], 'coord': result})

