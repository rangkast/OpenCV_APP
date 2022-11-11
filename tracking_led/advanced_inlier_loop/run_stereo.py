import matplotlib.pyplot as plt
import numpy as np
import qdarkstyle

from definition import *
from camera import *
import os
import sys
import usb.core
import usb.backend.libusb1
from uvc_openCV import *


def compare_length(a, b):
    if a >= b:
        return b
    else:
        return a


def get_points(blob_array):
    model_points = []
    image_points = []
    led_ids = []

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
        return ERROR, ERROR, ERROR

    return led_ids, np.array(model_points), np.array(image_points)


def stereo_calibrate(jdata):
    try:
        camid = jdata['stereol']['cam_id']

        led_num_l, points3D_l, points2D_l = get_points(jdata['stereol']['blobs'])
        led_num_r, points3D_r, points2D_r = get_points(jdata['stereor']['blobs'])

        # ToDo
        camera_k = leds_dic['cam_info'][camid]['cam_cal']['cameraK']
        dist_coeff = leds_dic['cam_info'][camid]['cam_cal']['dist_coeff']

        print('cam id ', camid)
        print('cam l info')
        print(led_num_l)
        print(points3D_l)
        print(points2D_l)
        print('cam r info')
        print(led_num_r)
        print(points3D_r)
        print(points2D_r)

        obj_points = []
        img_points_l = []
        img_points_r = []
        length = len(led_num_l)
        objectpoint = np.zeros((length, D3D), np.float32)
        imgpointl = np.zeros((length, D2D), np.float32)
        imgpointr = np.zeros((length, D2D), np.float32)

        for idx, led_num in enumerate(led_num_l):
            objectpoint[idx] = leds_dic['pts'][led_num]['pos']
            imgpointl[idx] = points2D_l[idx]
            imgpointr[idx] = points2D_r[idx]
        obj_points.append(objectpoint)
        img_points_l.append(imgpointl)
        img_points_r.append(imgpointr)
        print(obj_points)
        print(img_points_l)
        print(img_points_r)

        # # This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
        # retS, new_mtxL, distL, new_mtxR, distR, Rot, Trns, Emat, Fmat = cv2.stereoCalibrate(obj_pts, img_ptsL, img_ptsR,
        #                                                                                     new_mtxL, distL, new_mtxR,
        #                                                                                     distR,
        #                                                                                     imgL_gray.shape[::-1],
        #
        #                                                                                     criteria_stereo, flags)

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, CM1, dist0, CM2, dist1, R, T, E, F = cv2.stereoCalibrate(obj_points, img_points_l, img_points_r,
                                                                      camera_k, dist_coeff,
                                                                      camera_k, dist_coeff,
                                                                      (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT),
                                                                      criteria_stereo,
                                                                      flags)

        print(CM1, ' ', dist0)
        print(CM2, ' ', dist1)
        print('R ', R, '\nT ', T, '\nE ', E, '\nF ', F)

        rectify_scale = 1
        rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(CM1, dist0, CM2, dist1,
                                                                                  (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT), R, T,
                                                                                  rectify_scale, (0, 0))

        print('rect_l ', rect_l)
        print('rect_r ', rect_r)
        print('proj_mat_l ', proj_mat_l)
        print('proj_mat_r ', proj_mat_r)

        Left_Stereo_Map = cv2.initUndistortRectifyMap(CM1, dist0, rect_l, proj_mat_l,
                                                      (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT), cv2.CV_16SC2)
        Right_Stereo_Map = cv2.initUndistortRectifyMap(CM2, dist1, rect_r, proj_mat_r,
                                                       (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT), cv2.CV_16SC2)

        print("Saving paraeters ......")

        print(Left_Stereo_Map[0])
        cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_WRITE)
        cv_file.write("Left_Stereo_Map_x", Left_Stereo_Map[0])
        cv_file.write("Left_Stereo_Map_y", Left_Stereo_Map[1])
        cv_file.write("Right_Stereo_Map_x", Right_Stereo_Map[0])
        cv_file.write("Right_Stereo_Map_y", Right_Stereo_Map[1])
        cv_file.release()

        print('Q ', Q)
        json_file = ''.join(['jsons/test_result/', 'stereo.json'])
        group_data = OrderedDict()
        group_data['stereoRectify'] = {
            'rect_l': rect_l,
            'rect_r': rect_r,
            'proj_mat_l': proj_mat_l,
            'proj_mat_r': proj_mat_r,
            'Q': Q, 'roiL': roiL, 'roiR': roiR}

        rw_json_data(WRITE, json_file, group_data)

        return DONE, Left_Stereo_Map, Right_Stereo_Map

    except:
        return ERROR, NOT_SET, NOT_SET


if __name__ == '__main__':
    print('start stereo test main')
    cam_dev_list = terminal_cmd('v4l2-ctl', '--list-devices')

    leds_dic['test_status'] = init_test_status()
    leds_dic['cam_info'] = init_model_json(cam_dev_list)

    leds_dic['pts'] = init_coord_json(ORIGIN)
    leds_dic['target_pts'] = init_coord_json(TARGET)

    # load json file
    json_file = ''.join(['jsons/test_result/', f'{JSON_FILE}'])
    jdata = rw_json_data(READ, json_file, None)

    print('json data')
    print(jdata)

    ret, l_map, r_map = stereo_calibrate(jdata)

    if ret == DONE:
        cam_id = jdata['stereol']['cam_id']
        length = 2

        # 합성시킬 두 개의 영상 열기
        cap1 = cv2.VideoCapture(jdata['stereol']['file'])
        cap2 = cv2.VideoCapture(jdata['stereor']['file'])

        if not cap1.isOpened() or not cap2.isOpened():
            sys.exit()

        # 각 영상 프레임 수
        frame_cnt1 = round(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_cnt2 = round(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap1.get(cv2.CAP_PROP_FPS)
        effect_frames = int(fps * 2)

        delay = int(1000 / fps)

        print('frame_cnt1 ', frame_cnt1, ' frame_cnt2 ', frame_cnt2, ' ', )

        frame_count = compare_length(frame_cnt1, frame_cnt2)
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                while True:
                    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
                        cap1.release()
                        cap2.release()
                        cv2.destroyAllWindows()
                        break
                break
            else:

                imgL = frame1.copy()
                imgR = frame2.copy()
                # cv2.imshow("Left image before rectification", imgL)
                # cv2.imshow("Right image before rectification", imgR)
                Left_nice = cv2.remap(imgL, l_map[0], l_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
                Right_nice = cv2.remap(imgR, r_map[0], r_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
                # cv2.imshow("Left image after rectification", Left_nice)
                # cv2.imshow("Right image after rectification", Right_nice)
                # cv2.waitKey(0)
                #
                out = Right_nice.copy()
                out[:, :, 0] = Right_nice[:, :, 0]
                out[:, :, 1] = Right_nice[:, :, 1]
                out[:, :, 2] = Left_nice[:, :, 2]

                cv2.imshow("Output image", out)

                alpha = 0.5
                before_frame = cv2.addWeighted(frame1, alpha, frame2, alpha, 0)
                cv2.imshow('before frame', before_frame)
                #
                # after_frame = cv2.addWeighted(Left_nice, alpha, Right_nice, alpha, 0)
                # cv2.imshow('after frame', after_frame)

                cv2.waitKey(delay)

