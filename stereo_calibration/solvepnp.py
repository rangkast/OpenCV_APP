import datetime
import math
import numpy as np
from definition import *
import traceback
import copy
import cv2


def simple_solvePNP(cam_id, frame, blob_array):
    model_points = []
    image_points = []
    led_ids = []
    interationsCount = 100
    confidence = 0.99

    for blobs in blob_array:
        led_num = int(blobs['idx'])
        # if DEBUG == ENABLE:
        #     print('idx:', led_num, ' added 3d', leds_dic['pts'][led_num]['pos'], ' remake: ',
        #           leds_dic['target_pts'][led_num]['remake_3d'],
        #           ' 2d', [blobs['cx'], blobs['cy']])

        model_points.append(leds_dic['pts'][led_num]['pos'])
        led_ids.append(led_num)
        image_points.append([blobs['cx'], blobs['cy']])

    model_points_len = len(model_points)
    image_points_len = len(image_points)

    # check assertion
    if model_points_len != image_points_len:
        print("assertion len is not equal")
        return ERROR

    if model_points_len < 4 or image_points_len < 4:
        print("assertion < 4: ")
        return ERROR

    camera_k = leds_dic['cam_info'][cam_id]['cam_cal']['cameraK']
    dist_coeff = leds_dic['cam_info'][cam_id]['cam_cal']['dist_coeff']

    list_2d_distorted = np.zeros((image_points_len, 1, 2), dtype=np.float64)
    for i in range(image_points_len):
        list_2d_distorted[i] = image_points[i]

    points3D = np.array(model_points)

    list_2d_undistorted = cv2.fisheye.undistortPoints(list_2d_distorted, camera_k, dist_coeff)
    leds_dic['cam_info'][cam_id]['distorted_2d'] = copy.deepcopy(list_2d_distorted)
    leds_dic['cam_info'][cam_id]['undistorted_2d'] = copy.deepcopy(list_2d_undistorted)

    if DO_UNDISTORT == ENABLE:
        temp_points2D = []
        for u_data in list_2d_undistorted:
            temp_points2D.append([u_data[0][0], u_data[0][1]])
        points2D = np.array(temp_points2D)
        temp_camera_k = cameraK
        temp_dist_coeff = distCoeff
    else:
        points2D = np.array(image_points)
        temp_camera_k = leds_dic['cam_info'][cam_id]['cam_cal']['cameraK']
        temp_dist_coeff = leds_dic['cam_info'][cam_id]['cam_cal']['dist_coeff']

    if DO_SOLVEPNP_RANSAC == ENABLE:
        success, rvecs, tvecs, inliers = cv2.solvePnPRansac(points3D, points2D,
                                                            temp_camera_k,
                                                            temp_dist_coeff,
                                                            useExtrinsicGuess=True,
                                                            iterationsCount=interationsCount,
                                                            confidence=confidence,
                                                            reprojectionError=1.0,
                                                            flags=cv2.SOLVEPNP_ITERATIVE)

    else:
        success, rvecs, tvecs = cv2.solvePnP(points3D, points2D,
                                             temp_camera_k,
                                             temp_dist_coeff,
                                             flags=cv2.SOLVEPNP_AP3P)
        # length = len(points2D)
        # for i in range(length):
        #     inliers = np.array(
        #         [i for i in range(length)]).reshape(
        #         length, 1)

    # ret, RER, TRER, candidate_array, except_inlier = cal_RER_px(led_ids, frame,
    #                                                             points3D, points2D,
    #                                                             inliers,
    #                                                             rvecs,
    #                                                             tvecs,
    #                                                             temp_camera_k,
    #                                                             temp_dist_coeff, DONE)

    # if ret == SUCCESS:
    #     #####
    #     leds_dic['cam_info'][cam_id]['D_R_T_A'].append({'rvecs': rvecs, 'tvecs': tvecs})
    #     #####
    #     leds_dic['cam_info'][cam_id]['RER']['C_R_T'] = {'rvecs': rvecs, 'tvecs': tvecs}
    #     return SUCCESS, candidate_array
    # else:
    #     return ERROR, candidate_array

    leds_dic['cam_info'][cam_id]['D_R_T_A'].append({'rvecs': rvecs, 'tvecs': tvecs})
    #####
    leds_dic['cam_info'][cam_id]['RER']['C_R_T'] = {'rvecs': rvecs, 'tvecs': tvecs}
    return SUCCESS, blob_array


def cal_RER_px(led_ids, frame, points3D, points2D, inliers, rvecs, tvecs, camera_k, dist_coeff, status):
    # Compute re-projection error.
    blob_array = []
    points2D_reproj = cv2.projectPoints(points3D, rvecs,
                                        tvecs, camera_k, dist_coeff)[0].squeeze(1)
    # print('points2D_reproj\n', points2D_reproj, '\npoints2D\n', points2D, '\n inliers: ', inliers)
    assert (points2D_reproj.shape == points2D.shape)
    error = (points2D_reproj - points2D)[inliers]  # Compute error only over inliers.
    # print('error', error)
    rmse = 0.0
    dis = 0.0
    led_except = -1
    except_inlier = -1
    led_dis = []

    for idx, error_data in enumerate(error[:, 0]):
        rmse += np.power(error_data[0], 2) + np.power(error_data[1], 2)
        temp_dis = np.power(error_data[0], 2) + np.power(error_data[1], 2)
        led_dis.append(temp_dis)

        if status == NOT_SET:
            if temp_dis > dis:
                dis = temp_dis
                led_except = led_ids[idx]
                except_inlier = idx

        # print('led_num: ', led_ids[idx], ' dis:', '%0.18f' % temp_dis, ' : ',
        #       points2D_reproj[idx][0], ' ', points2D_reproj[idx][1],
        #       ' vs ', points2D[idx][0], ' ', points2D[idx][1])

        if temp_dis > 100:
            return ERROR, 0, 0, blob_array, except_inlier

    trmse = float(rmse - dis)
    # print('trmse : ', trmse, ' rmse : ', rmse)
    if inliers is None:
        return ERROR, -1, -1, blob_array, except_inlier
    RER = round(np.sqrt(rmse) / len(inliers), 18)
    if status == NOT_SET:
        TRER = round(np.sqrt(trmse) / (len(inliers) - 1), 18)
        if led_except == -1:
            return ERROR, RER, TRER, blob_array, except_inlier
    else:
        TRER = round(np.sqrt(trmse) / (len(inliers)), 18)

    for i, idx in enumerate(led_ids):
        if idx != led_except:
            blob_array.append({'idx': led_ids[i],
                               'cx': points2D[i][0], 'cy': points2D[i][1], 'area': 0})

    return SUCCESS, RER, TRER, blob_array, except_inlier


def cal_iqr_func(arr):
    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)

    IQR = Q3 - Q1

    outlier_step = 1.5 * IQR

    lower_bound = Q1 - outlier_step
    upper_bound = Q3 + outlier_step

    mask = np.where((arr > upper_bound) | (arr < lower_bound))

    # print(f"cal_iqr_func!!!!!! lower_bound = {lower_bound} upper_bound ={upper_bound} mask = {mask}")

    return mask


def detect_outliers(idx, blob_array, remove_index_array):
    temp_x = np.array(cal_iqr_func(blob_array[0]))
    temp_y = np.array(cal_iqr_func(blob_array[1]))

    for x in temp_x:
        for xx in x:
            if xx in remove_index_array:
                continue
            else:
                remove_index_array.append(xx)
    for y in temp_y:
        for yy in y:
            if yy in remove_index_array:
                continue
            else:
                remove_index_array.append(yy)

    remove_index_array.sort()

    # print("detect_outliers!!!!!!!!!!!!!!!!!!!!!!!!!!!! remove_index_array", remove_index_array)


def median_blobs(cam_id, blob_array, rt_array):
    blob_cnt = len(blob_array)
    if blob_cnt == 0:
        print('blob_cnt is 0')
        return ERROR
    blob_length = len(blob_array[0])

    med_blobs_array = []
    remove_index_array = []
    med_rt_array = []
    print('cam_id:', cam_id, ' blob_cnt:', blob_cnt)

    for i in range(blob_length):
        med_xy = [[], [], []]
        for ii in range(blob_cnt):
            med_xy[0].append(blob_array[ii][i]['cx'])
            med_xy[1].append(blob_array[ii][i]['cy'])
            # med_xy[2].append(blob_array[ii][i]['area'])
        detect_outliers(blob_array[ii][i]['idx'], med_xy, remove_index_array)

    r_len = len(remove_index_array)
    print(f"median_blobs!!!!! remove_index_array length={r_len}")

    for i in range(blob_length):
        med_xy = [[], [], []]
        for ii in range(blob_cnt):
            med_xy[0].append(blob_array[ii][i]['cx'])
            med_xy[1].append(blob_array[ii][i]['cy'])
            # med_xy[2].append(blob_array[ii][i]['area'])
        # tempx=med_xy[0]
        # print(f"original med_xy[0] = {tempx}")
        count = 0
        for index in remove_index_array:
            med_xy[0].pop(index - count)
            med_xy[1].pop(index - count)
            # med_xy[2].pop(index - count)
            count += 1
        # tempx=med_xy[0]
        # print(f"after pop med_xy[0] = {tempx}")

        mean_med_x = np.mean(med_xy[0])
        mean_med_y = np.mean(med_xy[1])

        med_blobs_array.append({'idx': blob_array[ii][i]['idx'],
                                'cx': mean_med_x,
                                'cy': mean_med_y})

    if rt_array != NOT_SET:
        count = 0
        for index in remove_index_array:
            rt_array.pop(index - count)
            count += 1
        for i in range(len(rt_array)):
            rvt = [[], [], []]
            for x in rt_array[i]['rvecs'][0]:
                rvt[0].append(x)
            for y in rt_array[i]['rvecs'][1]:
                rvt[1].append(y)
            for z in rt_array[i]['rvecs'][2]:
                rvt[2].append(z)
            tvt = [[], [], []]
            for x in rt_array[i]['tvecs'][0]:
                tvt[0].append(x)
            for y in rt_array[i]['tvecs'][1]:
                tvt[1].append(y)
            for z in rt_array[i]['tvecs'][2]:
                tvt[2].append(z)

        mean_rvt = []
        mean_tvt = []
        for i in range(0, 3):
            mean_rvt.append(np.mean(rvt[i]))
            mean_tvt.append(np.mean(tvt[i]))

        med_rt_array = {'rvecs': np.array([[mean_rvt[0]], [mean_rvt[1]], [mean_rvt[2]]], dtype=np.float64),
                        'tvecs': np.array([[mean_tvt[0]], [mean_tvt[1]], [mean_tvt[2]]], dtype=np.float64)}

        len_rt_array = len(rt_array)
    #     print(f"rt_array_len = {len_rt_array}")
    #     print(f"med_rt_array = {med_rt_array}")
    # print(f"med_blobs_array = {med_blobs_array}")

    blob_array = med_blobs_array

    return blob_array, med_rt_array
