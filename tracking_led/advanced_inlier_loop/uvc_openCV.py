import datetime
import math
import numpy as np
from definition import *
import traceback
import copy
from definition import *
from find_3d_coord import *

import cv2
from numba import jit


def adjust_boundary_filter(frame, cam_id):
    filter_array = leds_dic['cam_info'][cam_id]['filter']['filter_array']
    m_array = leds_dic['cam_info'][cam_id]['filter']['m_array']
    for h in range(m_array[0], m_array[2]):
        for w in range(m_array[1], m_array[3]):
            if filter_array[h][w] == -1:
                frame[h][w] = 0


def print_frame_data(frame, cam_id, blobs):
    for blob in blobs:
        if not blob['cx'] or not blob['cy']:
            continue

        cx = round(blob['cx'])
        cy = round(blob['cy'])
        # ncx = round(blob['newcx'])
        # ncy = round(blob['newcy'])
        # cv2.circle(frame, (cx, cy), 1, color=(0, 0, 255), thickness=-1)
        # draw red Point Cx, Cy

        # ToDo

        if PRINT_FRAME_INFOS == ENABLE:
            frame[cy][cx] = [0, 0, 255]
            cv2.circle(frame, (cx, cy), 15, color=(255, 255, 0), thickness=2)


def find_blob_center(cam_id, frame, blobs):
    filter_array = leds_dic['cam_info'][cam_id]['filter']['filter_array']
    m_array = leds_dic['cam_info'][cam_id]['filter']['m_array']
    lm_array = leds_dic['cam_info'][cam_id]['filter']['lm_array']
    spec = leds_dic['cam_info'][cam_id]['model']['spec']
    ret_blobs = []

    for blobs_spec in spec:
        led_num = int(blobs_spec['idx'])
        lm_spec = lm_array[led_num]

        # graySum 255 center
        x_sum = 0
        t_sum = 0
        y_sum = 0
        m_count = 0
        g_c_x = 0
        g_c_y = 0
        longa = (-1, -1)
        longb = (-1, -1)

        for y in range(lm_spec[0], lm_spec[2] + 1):
            for x in range(lm_spec[1], lm_spec[3] + 1):
                if frame[y][x] >= CV_MID_THRESHOLD:
                    x_sum += x * frame[y][x]
                    t_sum += frame[y][x]
                    m_count += 1

        for x in range(lm_spec[1], lm_spec[3] + 1):
            for y in range(lm_spec[0], lm_spec[2] + 1):
                if frame[y][x] >= CV_MID_THRESHOLD:
                    y_sum += y * frame[y][x]

        if t_sum != 0:
            g_c_x = x_sum / t_sum
            g_c_y = y_sum / t_sum

        if g_c_x == 0 or g_c_y == 0 or m_count > MAX_BLOB_SIZE:
            leds_dic['cam_info'][cam_id]['detect_status'][0] = NOT_SET
            return ERROR, ret_blobs

        ret_blobs.append({'idx': led_num, 'cx': g_c_x, 'cy': g_c_y, 'area': blobs_spec['size'],
                          'longa': longa, 'longb': longb})
    if USE_PRINT_FRAME == ENABLE:
        s = ""
        width_coord = 0
        for h in range(m_array[0], m_array[2]):
            if width_coord == 0:
                s = s + '%4s' % 'xy'
                for w in range(m_array[1], m_array[3]):
                    s = s + '%4d' % w
                s = s + '\n'
                width_coord = 1
            s = s + '%4d' % h
            for w in range(m_array[1], m_array[3]):
                s = s + '%4d' % frame[h][w]
            s = s + '\n'
        print(s)

    if len(spec) == len(ret_blobs):
        leds_dic['cam_info'][cam_id]['detect_status'][0] = DONE
        return DONE, ret_blobs
    else:
        leds_dic['cam_info'][cam_id]['detect_status'][0] = NOT_SET
        return ERROR, ret_blobs


def calc_frame_data(frame, cam_id, led_num, data):
    filter_array = leds_dic['cam_info'][cam_id]['filter']['filter_array']
    m_array = leds_dic['cam_info'][cam_id]['filter']['m_array']
    dy = [-1, 1, 0, 0]
    dx = [0, 0, -1, 1]
    for model in leds_dic['cam_info'][cam_id]['model']['spec']:
        if model['idx'] == led_num:
            print('need update ', cam_id, ' ', led_num, ' ', data)
            cx = int(model['x'])
            cy = int(model['y'])
            sadder = filter_array[cy][cx] + data
            queue = list()
            queue.append(np.array([cx, cy]))

            filter_array[cy][cx] = sadder
            frame[cy][cx][0] += data
            frame[cy][cx][1] += data
            frame[cy][cx][2] += data
            while queue:
                node = queue.pop(0)
                for i in range(4):
                    nx = np.int(node[0] + dx[i])
                    ny = np.int(node[1] + dy[i])
                    nadder = filter_array[ny][nx]
                    if m_array[0] < 0 or m_array[1] < 0 or ny > m_array[2] or nx > m_array[3]:
                        continue
                    if nadder != 0 and nadder != sadder:
                        filter_array[ny][nx] = sadder
                        new_frame_data = frame[ny][nx][2] + sadder
                        if new_frame_data < 0:
                            frame[ny][nx][0] = 0
                            frame[ny][nx][1] = 0
                            frame[ny][nx][2] = 0
                        else:
                            frame[ny][nx][0] = new_frame_data
                            frame[ny][nx][1] = new_frame_data
                            frame[ny][nx][2] = new_frame_data
                        queue.append(np.array([nx, ny]))
        leds_dic['cam_info'][cam_id]['filter']['filter_array'] = filter_array

    return


# ToDo
# @jit(nopython=True)
def calc_data(spec, cam_id):
    filtered_frame = [[-1] * CAP_PROP_FRAME_HEIGHT for i in range(CAP_PROP_FRAME_WIDTH)]
    filtered_lm_array = [0 for i in range(len(leds_dic['pts']))]
    dy = [-1, 1, 0, 0]
    dx = [0, 0, -1, 1]
    min_x = CAP_PROP_FRAME_WIDTH + 1
    min_y = CAP_PROP_FRAME_HEIGHT + 1
    max_x = -1
    max_y = -1
    try:
        for idx, s in enumerate(spec):
            queue = list()
            queue.append(s)
            # center coordinates
            cix = np.int(s[0])
            cx = np.int(s[1])
            cy = np.int(s[2])
            # 70% of json spec
            csize = np.int(s[3]) * 0.90
            # set visited
            # print('spec', s)

            filtered_frame[cy][cx] = cix

            # local min,max array
            lmin_x = CAP_PROP_FRAME_WIDTH + 1
            lmin_y = CAP_PROP_FRAME_HEIGHT + 1
            lmax_x = -1
            lmax_y = -1

            while queue:
                node = queue.pop(0)
                for i in range(4):
                    nx = np.int(node[1] + dx[i])
                    ny = np.int(node[2] + dy[i])
                    if ny < 0 or nx < 0 or ny > CAP_PROP_FRAME_HEIGHT or nx > CAP_PROP_FRAME_WIDTH:
                        continue
                    if np.sqrt(np.power((ny - cy), 2) + np.power((nx - cx), 2)) < csize and filtered_frame[ny][
                        nx] == -1:
                        filtered_frame[ny][nx] = cix
                        queue.append(np.array([cix, nx, ny, csize]))

                        if ny < min_y:
                            min_y = ny
                        if nx < min_x:
                            min_x = nx
                        if ny > max_y:
                            max_y = ny
                        if nx > max_x:
                            max_x = nx

                        if ny < lmin_y:
                            lmin_y = ny
                        if nx < lmin_x:
                            lmin_x = nx
                        if ny > lmax_y:
                            lmax_y = ny
                        if nx > lmax_x:
                            lmax_x = nx

            # end while
            filtered_lm_array[cix] = np.array([lmin_y, lmin_x, lmax_y, lmax_x])

        if min_y <= 10:
            min_y = 10
        if min_x <= 10:
            min_x = 10
        if max_y >= CAP_PROP_FRAME_HEIGHT - 10:
            max_y = CAP_PROP_FRAME_HEIGHT - 10
        if max_x >= CAP_PROP_FRAME_WIDTH - 10:
            max_x = CAP_PROP_FRAME_WIDTH - 10
        # print('min: ', min_x, ' ', min_y, ' max: ', max_x, ' ', max_y)

    except:
        print('exception')
        traceback.print_exc()
        return ERROR
    finally:
        leds_dic['cam_info'][cam_id]['filter'] = {'status': DONE, 'filter_array': filtered_frame,
                                                  'lm_array': filtered_lm_array,
                                                  'm_array': np.array([min_y, min_x, max_y, max_x])}
        if USE_PRINT_FRAME == ENABLE:
            print('cam_id:', cam_id, ' filter')
            s = ""
            for h in range(min_y, max_y):
                for w in range(min_x, max_x):
                    s = s + '%4d' % filtered_frame[h][w]
                s = s + '\n'
            print(s)

    return SUCCESS


def init_rt_custom_led_filter():
    for cam_id in range(len(leds_dic['cam_info'])):
        new_boundary = []
        boundary = leds_dic['cam_info'][cam_id]['model']['spec']
        for i, searching in enumerate(boundary):
            new_boundary.append(
                np.array([int(searching['idx']), int(searching['x']), int(searching['y']), int(searching['size'])]))

        if leds_dic['cam_info'][cam_id]['filter']['status'] == NOT_SET:
            if calc_data(np.array(new_boundary), cam_id) == SUCCESS:
                print(cam_id, ' DONE')
    return SUCCESS


def translate_led_id(frame, cam_id, blobs):
    blobs_detected = []
    boundary = leds_dic['cam_info'][cam_id]['model']['spec']
    expect_cnt = len(boundary)

    for coord in blobs:
        for searching in boundary:
            idx = searching['idx']
            bx = searching['x']
            by = searching['y']
            size = searching['size']

            coord_size = math.sqrt(math.pow(coord['cx'] - bx, 2) + math.pow(coord['cy'] - by, 2))
            if coord_size < size and (LED_AREA_MIN < coord['area'] < LED_AREA_MAX):
                coord['idx'] = idx
                blobs_detected.append({'idx': idx, 'cx': coord['cx'], 'cy': coord['cy'], 'area': coord['area'],
                                       'longa': coord['longa'], 'longb': coord['longb'], 'ecc': coord['ecc']})

    if USE_CUSTOM_FILTER == DISABLE:
        if len(blobs_detected) == expect_cnt:
            leds_dic['cam_info'][cam_id]['detect_status'][0] = DONE
        else:
            leds_dic['cam_info'][cam_id]['detect_status'][0] = NOT_SET

    return leds_dic['cam_info'][cam_id]['detect_status'][0], blobs_detected


def rt_contour_detect(frame, cam_id, cv_threshold):
    blob_array = []
    color_list = [(238, 0, 0), (0, 252, 124), (142, 56, 142)]

    if USE_CUSTOM_FILTER == ENABLE:
        ret, img_contour_binary = cv2.threshold(frame, CV_FINDCONTOUR_LVL, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
        img_for_contour_gray = cv2.cvtColor(img_contour_binary, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(img_for_contour_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ret, img_binary = cv2.threshold(frame, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
        img_gray = cv2.cvtColor(img_binary, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, img_binary = cv2.threshold(img_gray, 30, 255, 0)
        contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        longa = (-1, -1)
        longb = (-1, -1)

        if M["m00"] == 0:  # this is a line
            shape = "line"
        else:
            cx = float(M['m10'] / M['m00'])
            cy = float(M['m01'] / M['m00'])
            eccentricity = 0
            if len(cnt) >= 5:  # should be at least 5 points to fit the ellipse
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                a = ma / 2
                b = MA / 2
                eccentricity = math.sqrt(pow(a, 2) - pow(b, 2))
                eccentricity = round(eccentricity / a, 2)

            blob_array.append(
                {'idx': idx, 'cx': cx, 'cy': cy, 'area': area, 'longa': longa, 'longb': longb, 'ecc': eccentricity})
            # cv2.circle(img_gray, (int(cx), int(cy)), 1, color=(255, 255, 255), thickness=-1)
            # if cam_id == 0:
            # cv2.putText(frame, f'{idx}', (cx + 10, cy + 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_4)
            #     print('idx:', idx, ' cx:', cx, ' cy:', cy)

    return img_gray, blob_array


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
            med_xy[2].append(blob_array[ii][i]['area'])
        detect_outliers(blob_array[ii][i]['idx'], med_xy, remove_index_array)

    r_len = len(remove_index_array)
    print(f"median_blobs!!!!! remove_index_array length={r_len}")

    for i in range(blob_length):
        med_xy = [[], [], []]
        for ii in range(blob_cnt):
            med_xy[0].append(blob_array[ii][i]['cx'])
            med_xy[1].append(blob_array[ii][i]['cy'])
            med_xy[2].append(blob_array[ii][i]['area'])
        # tempx=med_xy[0]
        # print(f"original med_xy[0] = {tempx}")
        count = 0
        for index in remove_index_array:
            med_xy[0].pop(index - count)
            med_xy[1].pop(index - count)
            med_xy[2].pop(index - count)
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


# NON-Planar calibration  안됨.
def calibrate_camera(cam_id, frame, blob_array):
    print('calibrate camera')
    led_ids = []
    obj_points = []
    img_points = []
    length = len(blob_array)
    objectpoint = np.zeros((length, D3D), np.float32)
    imgpoint = np.zeros((length, D2D), np.float32)

    for idx, blobs in enumerate(blob_array):
        led_num = int(blobs['idx'])
        objectpoint[idx] = leds_dic['pts'][led_num]['pos']
        led_ids.append(led_num)
        imgpoint[idx] = [blobs['cx'], blobs['cy']]

    obj_points.append(objectpoint)
    img_points.append(imgpoint)
    print(obj_points)
    print(img_points)

    # Calibrating left camera

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, frame.shape[::-1], None, None)
    h, w = frame.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    if DEBUG == ENABLE:
        print('mtx:', mtx, ' dist:', dist, ' rvecs:', rvecs, ' tvecs:', tvecs)
        print('h:', h, ' w:', w)
        print('new_mtx:', new_mtx, ' roi:', roi)

    return


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
        length = len(points2D)
        for i in range(length):
            inliers = np.array(
                [i for i in range(length)]).reshape(
                length, 1)

    ret, RER, TRER, candidate_array, except_inlier = cal_RER_px(led_ids, frame,
                                                                points3D, points2D,
                                                                inliers,
                                                                rvecs,
                                                                tvecs,
                                                                temp_camera_k,
                                                                temp_dist_coeff, DONE)

    if DO_SOLVEPNP_REFINE == ENABLE:
        if ret == SUCCESS:
            # print('RER before LM refinement: ', str(RER), ' ', str(TRER))
            # Refine estimate using LM
            if not success:
                print('Initial estimation unsuccessful, skipping refinement')
            elif not hasattr(cv2, 'solvePnPRefineLM'):
                print('solvePnPRefineLM requires OpenCV >= 4.1.1, skipping refinement')
            else:
                assert len(inliers) >= 3, 'LM refinement requires at least 3 inlier points'
                new_points3D = []
                new_points2D = []
                new_led_ids = []
                for inlier_num in range(0, len(inliers)):
                    if inlier_num != except_inlier:
                        new_points3D.append(model_points[inlier_num])
                        new_points2D.append(points2D[inlier_num])
                        new_led_ids.append(led_ids[inlier_num])
                        # print(inlier_num, ' 3D ', points3D[inlier_num], ' 2D ', points2D[inlier_num])
                # print('except inlier ', except_inlier, ' inliers len ', len(inliers), ' ', new_inlier)

                inliers_loop_array = []
                loop_RER = 0
                length = len(inliers) - 1
                for i in range(len(inliers)):
                    new_inlier = np.array(
                        [i for i in range(length)]).reshape(
                        length, 1)
                    # 0, 1, 2, 3
                    cnt = 0
                    index_cnt = 0
                    while cnt < 4:
                        if cnt != i:
                            new_inlier[index_cnt][0] = cnt
                            index_cnt += 1
                        cnt += 1
                    # print('i ', i, new_inlier)
                    cv2.solvePnPRefineLM(points3D[new_inlier],
                                         points2D[new_inlier], temp_camera_k, temp_dist_coeff,
                                         rvecs, tvecs)
                    # print('after LM ransac rvecs ', rvecs[:, 0], ' tvecs ', tvecs[:, 0])
                    lm_rvecs = copy.deepcopy(rvecs)
                    lm_tvecs = copy.deepcopy(tvecs)
                    ret, RER, TRER, lm_candidate_array, except_inlier = cal_RER_px(led_ids, frame,
                                                                                   points3D, points2D,
                                                                                   inliers,
                                                                                   lm_rvecs,
                                                                                   lm_tvecs,
                                                                                   temp_camera_k,
                                                                                   temp_dist_coeff, DONE)
                    loop_RER += RER
                    inliers_loop_array.append({'cam_id': cam_id, 'inliers': new_inlier,
                                               'candidates': lm_candidate_array,
                                               'rvecs': lm_rvecs,
                                               'tvecs': lm_tvecs,
                                               'rer': RER})

                if loop_RER < leds_dic['cam_info'][cam_id]['RER']['loop']:
                    leds_dic['cam_info'][cam_id]['inliers_loop'] = inliers_loop_array
                    leds_dic['cam_info'][cam_id]['RER']['loop'] = loop_RER
                    # print(inliers_loop_array)

                # print('RER after LM refinement: ', str(RER), ' ', str(TRER))

    if ret == SUCCESS:
        #####
        leds_dic['cam_info'][cam_id]['D_R_T_A'].append({'rvecs': rvecs, 'tvecs': tvecs})
        #####
        leds_dic['cam_info'][cam_id]['RER']['C_R_T'] = {'rvecs': rvecs, 'tvecs': tvecs}
        return SUCCESS, candidate_array
    else:
        return ERROR, candidate_array

