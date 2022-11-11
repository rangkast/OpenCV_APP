import time
from collections import deque

import cv2
import imutils
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import threading

from PyQt5.QtWidgets import QLabel

from main import *
from definition import *
from collections import OrderedDict
from solvepnp import *


def find_center(frame, led_num, X, Y, W, H, blobs):
    x_sum = 0
    t_sum = 0
    y_sum = 0
    m_count = 0
    g_c_x = 0
    g_c_y = 0

    ret_blobs = copy.deepcopy(blobs)

    for y in range(Y, Y + H):
        for x in range(X, X + W):
            if frame[y][x] >= CV_MID_THRESHOLD:
                x_sum += x * frame[y][x]
                t_sum += frame[y][x]
                m_count += 1

    for x in range(X, X + W):
        for y in range(Y, Y + H):
            if frame[y][x] >= CV_MID_THRESHOLD:
                y_sum += y * frame[y][x]

    if t_sum != 0:
        g_c_x = x_sum / t_sum
        g_c_y = y_sum / t_sum

    # print('led ', led_num, ' x ', g_c_x, ' y ', g_c_y)

    if g_c_x == 0 or g_c_y == 0:
        return ERROR

    if len(ret_blobs) > 0:
        detect = 0
        for i, datas in enumerate(ret_blobs):
            led = datas['idx']
            if led == led_num:
                ret_blobs[i] = {'idx': led_num, 'cx': g_c_x, 'cy': g_c_y}
                detect = 1
                break
        if detect == 0:
            ret_blobs.append({'idx': led_num, 'cx': g_c_x, 'cy': g_c_y})
    else:
        ret_blobs.append({'idx': led_num, 'cx': g_c_x, 'cy': g_c_y})

    return DONE, ret_blobs


def view_camera_infos(frame, text, x, y):
    cv2.putText(frame, text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), lineType=cv2.LINE_AA)


trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.legacy.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


def camera_rt_test():
    bboxes = []
    blobs = []

    for cam_id in range(len(leds_dic['cam_info'])):
        print('try to open:', leds_dic['cam_info'][cam_id]['port'])
        video_src = leds_dic['cam_info'][cam_id]['port']

        cap = cv2.VideoCapture(video_src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FORMAT, cv2.CV_64FC1)

        while True:
            ret, frame = cap.read()
            if not ret:
                print('Cannot read video file')
                break

            img_draw = frame.copy()

            if cv2.waitKey(1) == ord('a'):
                view_camera_infos(frame, 'drag led area and press space bar',
                                  30, 35)
                cv2.imshow('MultiTracker', frame)
                bbox = cv2.selectROI('MultiTracker', frame)
                print("Press q to quit selecting boxes and start tracking")
                print("Press any other key to select next object")
                view_camera_infos(frame, 'press led numer',
                                  30, 70)
                cv2.imshow('MultiTracker', frame)
                while True:
                    # ToDo 수정해야 함
                    key = cv2.waitKey(1) & 0xff
                    if key in range(48, 58):  # 0~9 숫자 입력   ---⑥
                        IDX = key - 48  # 선택한 숫자로 트랙커 인덱스 수정
                        print('led num ', IDX)
                        bboxes.append({'idx': IDX, 'bbox': bbox})
                        break
                    elif cv2.waitKey(1) == ord('q'):
                        bboxes.clear()
                        break

            elif cv2.waitKey(1) == ord('n'):
                break

            elif cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
                cap.release()
                cv2.destroyAllWindows()
                return

            if len(bboxes) > 0:
                for i, data in enumerate(bboxes):
                    (x, y, w, h) = data['bbox']
                    IDX = data['idx']
                    cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
                    view_camera_infos(img_draw, ''.join([f'{IDX}']), int(x), int(y) - 10)
                    view_camera_infos(img_draw, ''.join(['[', f'{IDX}', '] '
                                                            , f' {x}'
                                                            , f' {y}']), 30, 35 + i * 30)

            view_camera_infos(img_draw, ''.join(['Cam[', f'{cam_id}', '] ', f'{video_src}']),
                              CAP_PROP_FRAME_WIDTH - 250, 35)

            cv2.circle(img_draw, (int(CAP_PROP_FRAME_WIDTH / 2), int(CAP_PROP_FRAME_HEIGHT / 2)), 2, color=(0, 0, 255),
                       thickness=-1)

            cv2.imshow('MultiTracker', img_draw)

        # end while
        print('Selected bounding boxes {}'.format(bboxes))

        # Specify the tracker type
        trackerType = "CSRT"

        # Create MultiTracker object
        multiTracker = cv2.legacy.MultiTracker_create()

        tracker_start = 0
        recording_start = 0

        # Process video and track objects
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            ret, img_contour_binary = cv2.threshold(frame, CV_FINDCONTOUR_LVL, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
            img_gray = cv2.cvtColor(img_contour_binary, cv2.COLOR_BGR2GRAY)

            # Initialize MultiTracker
            if tracker_start == 0:
                for i, data in enumerate(bboxes):
                    multiTracker.add(createTrackerByName(trackerType), img_gray, data['bbox'])

            tracker_start = 1

            # get updated location of objects in subsequent frames
            qq, boxes = multiTracker.update(img_gray)

            # draw tracked objects
            for i, newbox in enumerate(boxes):
                p1 = (int(newbox[0]), int(newbox[1]))
                p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                cv2.rectangle(img_gray, p1, p2, 255, 1)
                IDX = bboxes[i]['idx']
                view_camera_infos(img_gray, ''.join([f'{IDX}']), int(newbox[0]), int(newbox[1]) - 10)

            KEY = cv2.waitKey(1)

            # add graysum find center
            for i, newbox in enumerate(boxes):
                IDX = bboxes[i]['idx']
                ret, new_blobs = find_center(img_gray, IDX, int(newbox[0]), int(newbox[1]),
                                             int(newbox[2]), int(newbox[3]), blobs)
                if ret == DONE:
                    blobs = new_blobs

            # ToDo
            if ret == DONE:
                ret_status, min_blob = simple_solvePNP(cam_id, frame, blobs)
                if ret_status == SUCCESS:
                    leds_dic['cam_info'][cam_id]['blobs'] = min_blob
                    leds_dic['cam_info'][cam_id]['med_blobs'].append(min_blob)

                    cv2.rectangle(img_gray, (10, 10), (CAP_PROP_FRAME_WIDTH - 10, CAP_PROP_FRAME_HEIGHT - 10), 255, 2)
                    cam_ori = R.from_rotvec(leds_dic['cam_info'][cam_id]['RER']['C_R_T']['rvecs'].reshape(3)).as_quat()
                    cam_ori_euler = np.round_(get_euler_from_quat('xyz', cam_ori), 3)
                    cam_ori_quat = np.round_(get_quat_from_euler('xyz', cam_ori_euler), 8)
                    cam_pos = leds_dic['cam_info'][cam_id]['RER']['C_R_T']['tvecs'].reshape(3)
                    cv2.putText(img_gray, ''.join(['rot 'f'{cam_ori_euler}']),
                                (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, lineType=cv2.LINE_AA)
                    cv2.putText(img_gray, ''.join(['quat 'f'{cam_ori_quat}']),
                                (20, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, lineType=cv2.LINE_AA)
                    cv2.putText(img_gray, ''.join(['pos 'f'{cam_pos}']),
                                (20, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, lineType=cv2.LINE_AA)
                stacked = len(leds_dic['cam_info'][cam_id]['med_blobs'])
                cv2.putText(img_gray, ''.join([f'{stacked}', ' data stacked']),
                            (20, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, lineType=cv2.LINE_AA)

            # quit on ESC button
            if KEY & 0xFF == 27:  # Esc pressed
                cap.release()
                cv2.destroyAllWindows()
                return
            elif KEY == ord('e'):
                break
            elif KEY == ord('s'):
                if recording_start == 0:
                    # w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    # h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    # print('w', w, 'h', h, 'fps', fps)
                    # fourcc  val 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
                    delay = round(1000 / fps)
                    now = datetime.datetime.now()
                    recording_file_name = ''.join([f'{now}', '.avi'])
                    print('recording start', ' ', recording_file_name)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    recording_out = cv2.VideoWriter(recording_file_name, fourcc, fps,
                                                    (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT))
                    recording_start = 1

                    leds_dic['cam_info'][cam_id]['track_cal']['recording'] = {'name': recording_file_name}

            elif KEY == ord('c'):
                blob_array, rt_array = median_blobs(cam_id,
                                                    leds_dic['cam_info'][cam_id]['med_blobs'],
                                                    leds_dic['cam_info'][cam_id]['D_R_T_A'])

                leds_dic['cam_info'][cam_id]['track_cal']['data'] = {'idx': cam_id, 'blobs': blob_array,
                                                                     'R_T': rt_array}

                leds_dic['cam_info'][cam_id]['med_blobs'].clear()
                leds_dic['cam_info'][cam_id]['D_R_T_A'].clear()

                print(leds_dic['cam_info'][cam_id]['track_cal'])

                if recording_start == 1:
                    recording_out.release()
                    recording_start = 0
                break

            # print current stacked data
            for i, track_data in enumerate(leds_dic['cam_info'][cam_id]['track_cal']['data']):
                track_r = R.from_rotvec(track_data['R_T']['rvecs'].reshape(3)).as_quat()
                track_r_euler = np.round_(get_euler_from_quat('xyz', track_r), 3)
                track_t = track_data['R_T']['tvecs'].reshape(3)
                cv2.putText(img_gray, ''.join(['R 'f'{track_r_euler}',
                                               ' T 'f'{track_t}']),
                            (CAP_PROP_FRAME_WIDTH - 400, 35 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, lineType=cv2.LINE_AA)

            if recording_start == 1:
                recording_out.write(frame)

            # show frame
            cv2.imshow('MultiTracker', img_gray)

            # release camera frame
        if recording_start == 1:
            recording_out.release()

        cap.release()
        cv2.destroyAllWindows()

        bboxes.clear()
        blobs.clear()


def rectification(imgL, imgR, l_map, r_map):
    img_left = imgL.copy()
    img_right = imgR.copy()
    Left_nice = cv2.remap(img_left, l_map[0], l_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(img_right, r_map[0], r_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    return Left_nice, Right_nice


def disparity_map(imgL, imgR):
    window_size = 3

    left_matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=160,
                                         blockSize=25,
                                         P1=8 * 3 * window_size ** 2,
                                         P2=32 * 3 * window_size ** 2,
                                         disp12MaxDiff=1,
                                         uniquenessRatio=15,
                                         speckleWindowSize=0,
                                         speckleRange=2,
                                         preFilterCap=63,
                                         mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                         )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    lmbda = 80000
    sigma = 1.2
    visual_multiplyer = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    disp_left = left_matcher.compute(imgL, imgR)
    disp_right = right_matcher.compute(imgR, imgL)
    disp_left = np.int16(disp_left)
    disp_right = np.int16(disp_right)

    disp = wls_filter.filter(disp_left, imgL, None, disp_right)
    disp = cv2.normalize(src=disp, dst=disp,
                         beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    disp = np.uint8(disp)

    return disp


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def make_point_3d(jdata, disp, imgL, imgR):
    Q = np.float32([jdata['stereoRectify']['Q'][0],
                    jdata['stereoRectify']['Q'][1],
                    jdata['stereoRectify']['Q'][2],
                    jdata['stereoRectify']['Q'][3]])
    points = cv2.reprojectImageTo3D(disp, Q)
    reflect_matrix = np.identity(3)
    reflect_matrix[0] *= -1
    points = np.matmul(points, reflect_matrix)

    # extract colors from image
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    # filter by min disparity
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]

    # filter by dimension
    idx = np.fabs(out_points[:, 0]) < 4.5
    out_points = out_points[idx]
    out_colors = out_colors.reshape(-1, 3)
    out_colors = out_colors[idx]

    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')

    reflected_pts = np.matmul(out_points, reflect_matrix)
    projected_img, _ = cv2.projectPoints(reflected_pts, np.identity(3), np.array([0., 0., 0.]),
                                         leds_dic['cam_info'][1]['cam_cal']['cameraK'],
                                         leds_dic['cam_info'][1]['cam_cal']['dist_coeff'])

    projected_img = projected_img.reshape(-1, 2)
    blank_img = np.zeros(imgL.shape, 'uint8')
    img_colors = imgR[mask][idx].reshape(-1, 3)

    for i, pt in enumerate(projected_img):
        pt_x = int(pt[0])
        pt_y = int(pt[1])
        if pt_x > 0 and pt_y > 0:
            # use the BGR format to match the original image type
            col = (int(img_colors[i, 2]), int(img_colors[i, 1]), int(img_colors[i, 0]))
            cv2.circle(blank_img, (pt_x, pt_y), 1, col)

    # h, w = imgL.shape[:2]
    # f = 0.8 * w  # guess for focal length
    # Q = np.float32([[1, 0, 0, -0.5 * w],
    #                 [0, -1, 0, 0.5 * h],  # turn points 180 deg around x-axis,
    #                 [0, 0, 0, -f],  # so that y-axis looks up
    #                 [0, 0, 1, 0]])
    # points = cv2.reprojectImageTo3D(disp, Q)
    # colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    # mask = disp > disp.min()
    # out_points = points[mask]
    # out_colors = colors[mask]
    # out_fn = 'out.ply'
    # write_ply(out_fn, out_points, out_colors)
    # print('%s saved' % out_fn)

    return blank_img


def stereo_camera_start():
    print('start open_camera')
    ret, l_map, r_map = rw_file_storage(READ, NOT_SET, NOT_SET)
    # load json file
    json_file = ''.join(['jsons/test_result/', f'{JSON_FILE}'])
    jdata = rw_json_data(READ, json_file, None)

    if ret == DONE:
        cap1 = cv2.VideoCapture(leds_dic['cam_info'][0]['port'])
        cap2 = cv2.VideoCapture(leds_dic['cam_info'][1]['port'])

        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)
        # cap1.set(cv2.CAP_PROP_FORMAT, cv2.CV_64FC1)

        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)
        # cap2.set(cv2.CAP_PROP_FORMAT, cv2.CV_64FC1)

        if not cap1.isOpened() or not cap2.isOpened():
            sys.exit()
        fps = cap1.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps)

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                break
            imgL = frame1.copy()
            imgR = frame2.copy()

            KEY = cv2.waitKey(1)

            # show origin frame
            alpha = 0.5
            origin_frame = cv2.addWeighted(imgL, alpha, imgR, alpha, 0)
            # cv2.imshow('origin frame', origin_frame)

            # Rectification
            R_L_F, R_R_F = rectification(imgL, imgR, l_map, r_map)
            out = R_R_F.copy()
            out[:, :, 0] = R_R_F[:, :, 0]
            out[:, :, 1] = R_R_F[:, :, 1]
            out[:, :, 2] = R_L_F[:, :, 2]
            cv2.imshow("rectification", out)

            # Disparity
            disparity_frame = disparity_map(R_L_F, R_R_F)
            cv2.imshow("disparity map", disparity_frame)

            if KEY == ord('c'):
                # make 3d point cloud
                point_cloud_frame = make_point_3d(jdata, disparity_frame, R_L_F, R_L_F)
                cv2.imshow("point cloud", point_cloud_frame)
            elif cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
                break



            cv2.waitKey(delay)

        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

        # Draw point cloud
        try:
            cloud = o3d.io.read_point_cloud("out.ply")  # Read the point cloud
            o3d.visualization.draw_geometries([cloud])
        except:
            print('error occured')


def camera_setting():
    print('start open_camera')

    cap1 = cv2.VideoCapture(leds_dic['cam_info'][0]['port'])
    cap1_name = leds_dic['cam_info'][0]['json']
    cap2 = cv2.VideoCapture(leds_dic['cam_info'][1]['port'])
    cap2_name = leds_dic['cam_info'][1]['json']

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)
    cap1.set(cv2.CAP_PROP_FORMAT, cv2.CV_64FC1)

    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)
    cap2.set(cv2.CAP_PROP_FORMAT, cv2.CV_64FC1)

    if not cap1.isOpened() or not cap2.isOpened():
        sys.exit()
    fps = cap1.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        imgL = frame1.copy()
        imgR = frame2.copy()
        view_camera_infos(imgL, f'{cap1_name}', 30, 35)
        cv2.circle(imgL, (int(CAP_PROP_FRAME_WIDTH / 2), int(CAP_PROP_FRAME_HEIGHT / 2)), 2, color=(0, 0, 255),
                   thickness=-1)
        cv2.imshow('left camera', imgL)
        view_camera_infos(imgR, f'{cap2_name}', 30, 35)
        cv2.circle(imgR, (int(CAP_PROP_FRAME_WIDTH / 2), int(CAP_PROP_FRAME_HEIGHT / 2)), 2, color=(0, 0, 255),
                   thickness=-1)
        cv2.imshow("right camera", imgR)

        alpha = 0.5

        after_frame = cv2.addWeighted(frame1, alpha, frame2, alpha, 0)
        cv2.imshow('stereo camera', after_frame)

        cv2.waitKey(delay)

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


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


def stereo_calibrate():
    try:
        led_num_l, points3D_l, points2D_l = get_points(leds_dic['cam_info'][0]['track_cal']['data']['blobs'])
        led_num_r, points3D_r, points2D_r = get_points(leds_dic['cam_info'][1]['track_cal']['data']['blobs'])

        # ToDo
        camera_k_l = leds_dic['cam_info'][0]['cam_cal']['cameraK']
        dist_coeff_l = leds_dic['cam_info'][0]['cam_cal']['dist_coeff']

        camera_k_r = leds_dic['cam_info'][1]['cam_cal']['cameraK']
        dist_coeff_r = leds_dic['cam_info'][1]['cam_cal']['dist_coeff']

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

        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, CM1, dist0, CM2, dist1, R, T, E, F = cv2.stereoCalibrate(obj_points, img_points_l, img_points_r,
                                                                      camera_k_l, dist_coeff_l,
                                                                      camera_k_r, dist_coeff_r,
                                                                      (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT),
                                                                      criteria_stereo,
                                                                      flags)

        print(CM1, ' ', dist0)
        print(CM2, ' ', dist1)
        print('R ', R, '\nT ', T, '\nE ', E, '\nF ', F)

        rectify_scale = 1
        rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(CM1, dist0, CM2, dist1,
                                                                                  (CAP_PROP_FRAME_WIDTH,
                                                                                   CAP_PROP_FRAME_HEIGHT),
                                                                                  R, T,
                                                                                  rectify_scale, (0, 0))

        print('rect_l ', rect_l)
        print('rect_r ', rect_r)
        print('proj_mat_l ', proj_mat_l)
        print('proj_mat_r ', proj_mat_r)

        Left_Stereo_Map = cv2.initUndistortRectifyMap(CM1, dist0, rect_l, proj_mat_l,
                                                      (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT), cv2.CV_16SC2)
        Right_Stereo_Map = cv2.initUndistortRectifyMap(CM2, dist1, rect_r, proj_mat_r,
                                                       (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT), cv2.CV_16SC2)

        print('save rectification map')
        rw_file_storage(WRITE, Left_Stereo_Map, Right_Stereo_Map)
        json_file = ''.join(['jsons/test_result/', f'{JSON_FILE}'])
        group_data = OrderedDict()
        group_data['stereoRectify'] = {
            'rect_l': rect_l.tolist(),
            'rect_r': rect_r.tolist(),
            'proj_mat_l': proj_mat_l.tolist(),
            'proj_mat_r': proj_mat_r.tolist(),
            'Q': Q.tolist()}
        group_data['stereol'] = {'cam_id': leds_dic['cam_info'][0]['json'],
                                 'blobs': leds_dic['cam_info'][0]['track_cal']['data']['blobs'],
                                 'rvecs': leds_dic['cam_info'][0]['track_cal']['data']['R_T']['rvecs'][:,
                                          0].tolist(),
                                 'tvecs': leds_dic['cam_info'][0]['track_cal']['data']['R_T']['tvecs'][:,
                                          0].tolist(),
                                 'file': leds_dic['cam_info'][0]['track_cal']['recording']['name']}
        group_data['stereor'] = {'cam_id': leds_dic['cam_info'][1]['json'],
                                 'blobs': leds_dic['cam_info'][1]['track_cal']['data']['blobs'],
                                 'rvecs': leds_dic['cam_info'][1]['track_cal']['data']['R_T']['rvecs'][:,
                                          0].tolist(),
                                 'tvecs': leds_dic['cam_info'][1]['track_cal']['data']['R_T']['tvecs'][:,
                                          0].tolist(),
                                 'file': leds_dic['cam_info'][1]['track_cal']['recording']['name']}
        print('save stereo info')
        rw_json_data(WRITE, json_file, group_data)

        return DONE, Left_Stereo_Map, Right_Stereo_Map
    except:
        traceback.print_exc()
        return ERROR, NOT_SET, NOT_SET
