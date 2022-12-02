# multi tracker test (PHONE)
import time
from datetime import datetime, date, time, timedelta

from definition import *

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


def set_camera_info():
    # camera setting test
    cap1_name = leds_dic['cam_info'][0]['port']
    cap1 = cv2.VideoCapture(cap1_name)
    cap2_name = leds_dic['cam_info'][1]['port']
    cap2 = cv2.VideoCapture(cap2_name)
    if SENSOR_NAME == 'Droidcam':
        width1 = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
        height1 = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print('cap1 size: %d, %d' % (width1, height1))
        leds_dic['cam_info'][0]['display']['width'] = width1
        leds_dic['cam_info'][0]['display']['height'] = height1

        width2 = cap2.get(cv2.CAP_PROP_FRAME_WIDTH)
        height2 = cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print('cap2 size: %d, %d' % (width2, height2))
        leds_dic['cam_info'][1]['display']['width'] = width2
        leds_dic['cam_info'][1]['display']['height'] = height2

        json_file = ''.join(['jsons/test_result/', f'{EXTERNAL_TOOL_CALIBRATION}'])
        external_jdata = rw_json_data(READ, json_file, None)
        if USE_EXTERNAL_TOOL_CALIBRAION == ENABLE:
            leds_dic['cam_info'][0]['cam_cal']['cameraK'] = np.array(external_jdata['stereol']['camera_k'],
                                                                     dtype=np.float64)
            leds_dic['cam_info'][0]['cam_cal']['dist_coeff'] = np.array(external_jdata['stereol']['distcoeff'],
                                                                        dtype=np.float64)
            leds_dic['cam_info'][1]['cam_cal']['cameraK'] = np.array(external_jdata['stereor']['camera_k'],
                                                                     dtype=np.float64)
            leds_dic['cam_info'][1]['cam_cal']['dist_coeff'] = np.array(external_jdata['stereor']['distcoeff'],
                                                                        dtype=np.float64)
        leds_dic['cam_info'][0]['display']['rotate'] = int(external_jdata['stereol']['rotate'])
        leds_dic['cam_info'][1]['display']['rotate'] = int(external_jdata['stereor']['rotate'])
    else:
        leds_dic['cam_info'][0]['display']['width'] = CAP_PROP_FRAME_WIDTH
        leds_dic['cam_info'][0]['display']['height'] = CAP_PROP_FRAME_HEIGHT
        leds_dic['cam_info'][1]['display']['width'] = CAP_PROP_FRAME_WIDTH
        leds_dic['cam_info'][1]['display']['height'] = CAP_PROP_FRAME_HEIGHT

    cap1.release()
    cap2.release()


# Not Used
def camera_setting_test():
    # camera setting test

    set_camera_info()

    cap1_name = leds_dic['cam_info'][0]['port']
    cap1 = cv2.VideoCapture(cap1_name)
    cap2_name = leds_dic['cam_info'][1]['port']
    cap2 = cv2.VideoCapture(cap2_name)

    width1 = leds_dic['cam_info'][0]['display']['width']
    height1 = leds_dic['cam_info'][0]['display']['height']

    width2 = leds_dic['cam_info'][1]['display']['width']
    height2 = leds_dic['cam_info'][1]['display']['height']

    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)
    cap1.set(cv2.CAP_PROP_FORMAT, cv2.CV_64FC1)

    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)
    cap2.set(cv2.CAP_PROP_FORMAT, cv2.CV_64FC1)

    if not cap1.isOpened() or not cap2.isOpened():
        sys.exit()

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break

        if SENSOR_NAME == 'Rift':
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        imgL = Rotate(frame1, int(leds_dic['cam_info'][0]['display']['rotate']))
        imgR = Rotate(frame2, int(leds_dic['cam_info'][1]['display']['rotate']))

        if SENSOR_NAME == 'Rift':
            cv2.circle(imgL, (int(width1 / 2), int(height1 / 2)), 3, color=(0, 0, 255),
                       thickness=-1)
            cv2.circle(imgR, (int(width2 / 2), int(height2 / 2)), 3, color=(0, 0, 255),
                       thickness=-1)
        else:
            cv2.circle(imgL, (int(height1 / 2), int(width1 / 2)), 3, color=(0, 0, 255),
                       thickness=-1)
            cv2.circle(imgR, (int(height2 / 2), int(width2 / 2)), 3, color=(0, 0, 255),
                       thickness=-1)
        view_camera_infos(imgL, f'{cap1_name}', 30, 35)
        cv2.imshow('left camera', imgL)
        view_camera_infos(imgR, f'{cap2_name}', 30, 35)
        cv2.imshow("right camera", imgR)

        cv2.waitKey(CAM_DELAY)

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


def pairPoints(key, target):
    print('start ', pairPoints.__name__)
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


def ret_key(rt_status):
    if rt_status == DYNAMIC_RT:
        rt_key = 'D_R_T'
    elif rt_status == STATIC_RT:
        rt_key = 'S_R_T'
    elif rt_status == DYNAMIC_RT_MED:
        rt_key = 'D_R_T_A'
    elif rt_status == DYNAMIC_RT_QUIVER:
        rt_key = 'D_R_T_Q'
    elif rt_status == MED_RT:
        rt_key = 'M_R_T'
    else:
        rt_key = NOT_SET

    return rt_key


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
                    {'idx': i, 'cam_l': data[0]['cidx'], 'cam_r': data[1]['cidx'], 'coord': result})


def draw_ax_plot():
    plt.style.use('default')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # draw origin data
    draw_dots(D3D, leds_dic['target_pts'], ax, 'blue')
    draw_dots(D3DT, leds_dic['target_pts'], ax, 'gray')

    # Add refactoring data

    # 원점
    ax.scatter(0, 0, 0, marker='o', color='k', s=20)
    ax.set_xlim([-0.5, 0.5])
    ax.set_xlabel('X')
    ax.set_ylim([-0.5, 0.5])
    ax.set_ylabel('Y')
    ax.set_zlim([-0.5, 0.5])
    ax.set_zlabel('Z')
    scale = 1.5
    f = zoom_factory(ax, base_scale=scale)
    plt.show()


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

# ToDo
# def stereo_calibrate():
#     try:
#         led_num_l, points3D_l, points2D_l = get_points(leds_dic['cam_info'][0]['med_blobs'])
#         led_num_r, points3D_r, points2D_r = get_points(leds_dic['cam_info'][1]['med_blobs'])
#
#         # ToDo
#         camera_k_l = leds_dic['cam_info'][0]['cam_cal']['cameraK']
#         dist_coeff_l = leds_dic['cam_info'][0]['cam_cal']['dist_coeff']
#
#         camera_k_r = leds_dic['cam_info'][1]['cam_cal']['cameraK']
#         dist_coeff_r = leds_dic['cam_info'][1]['cam_cal']['dist_coeff']
#
#         print('cam l info')
#         print(led_num_l)
#         print(points3D_l)
#         print(points2D_l)
#         print('cam r info')
#         print(led_num_r)
#         print(points3D_r)
#         print(points2D_r)
#
#         obj_points = []
#         img_points_l = []
#         img_points_r = []
#         length = len(led_num_l)
#         objectpoint = np.zeros((length, D3D), np.float32)
#         imgpointl = np.zeros((length, D2D), np.float32)
#         imgpointr = np.zeros((length, D2D), np.float32)
#
#         for idx, led_num in enumerate(led_num_l):
#             objectpoint[idx] = leds_dic['pts'][led_num]['pos']
#             imgpointl[idx] = points2D_l[idx]
#             imgpointr[idx] = points2D_r[idx]
#         obj_points.append(objectpoint)
#         img_points_l.append(imgpointl)
#         img_points_r.append(imgpointr)
#         print(obj_points)
#         print(img_points_l)
#         print(img_points_r)
#
#         flags = 0
#         flags |= cv2.CALIB_FIX_INTRINSIC
#         criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#         ret, CM1, dist0, CM2, dist1, R, T, E, F = cv2.stereoCalibrate(obj_points, img_points_l, img_points_r,
#                                                                       camera_k_l, dist_coeff_l,
#                                                                       camera_k_r, dist_coeff_r,
#                                                                       (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT),
#                                                                       criteria_stereo,
#                                                                       flags)
#         print('CM1 , dist0')
#         print(CM1, ' ', dist0)
#         print('CM2 , dist1')
#         print(CM2, ' ', dist1)
#         print('R ', R, '\nT ', T, '\nE ', E, '\nF ', F)
#
#         rectify_scale = 1
#         rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(CM1, dist0, CM2, dist1,
#                                                                                   (CAP_PROP_FRAME_WIDTH,
#                                                                                    CAP_PROP_FRAME_HEIGHT),
#                                                                                   R, T,
#                                                                                   rectify_scale, (0, 0))
#
#         print('rect_l ', rect_l)
#         print('rect_r ', rect_r)
#         print('proj_mat_l ', proj_mat_l)
#         print('proj_mat_r ', proj_mat_r)
#
#         Left_Stereo_Map = cv2.initUndistortRectifyMap(CM1, dist0, rect_l, proj_mat_l,
#                                                       (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT), cv2.CV_16SC2)
#         Right_Stereo_Map = cv2.initUndistortRectifyMap(CM2, dist1, rect_r, proj_mat_r,
#                                                        (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT), cv2.CV_16SC2)
#
#         return DONE, Left_Stereo_Map, Right_Stereo_Map
#     except:
#         traceback.print_exc()
#         return ERROR, NOT_SET, NOT_SET
#


def camera_rt_test():
    set_camera_info()

    bboxes = []
    blobs = []

    for cam_id in range(len(leds_dic['cam_info'])):
        print('try to open:', leds_dic['cam_info'][cam_id]['port'])
        video_src = leds_dic['cam_info'][cam_id]['port']

        width = int(leds_dic['cam_info'][cam_id]['display']['width'])
        height = int(leds_dic['cam_info'][cam_id]['display']['height'])

        cap = cv2.VideoCapture(video_src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FORMAT, cv2.CV_64FC1)

        print('width ', leds_dic['cam_info'][cam_id]['display']['width'])
        print('height ', leds_dic['cam_info'][cam_id]['display']['height'])

        while True:
            ret, frame = cap.read()
            if not ret:
                print('Cannot read video file')
                break
            if SENSOR_NAME == 'Rift':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = Rotate(frame, int(leds_dic['cam_info'][cam_id]['display']['rotate']))
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
                break

            if len(bboxes) > 0:
                for i, data in enumerate(bboxes):
                    (x, y, w, h) = data['bbox']
                    IDX = data['idx']
                    cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 2, 1)
                    view_camera_infos(img_draw, ''.join([f'{IDX}']), int(x), int(y) - 10)
                    view_camera_infos(img_draw, ''.join(['[', f'{IDX}', '] '
                                                            , f' {x}'
                                                            , f' {y}']), 30, 35 + i * 30)

            if SENSOR_NAME == 'Rift':
                view_camera_infos(img_draw, ''.join(['Cam[', f'{cam_id}', '] ', f'{video_src}']),
                                  width - 250, 35)
            else:
                view_camera_infos(img_draw, ''.join(['Cam[', f'{cam_id}', '] ', f'{video_src}']),
                                  height - 500, 35)

            if SENSOR_NAME == 'Rift':
                cv2.circle(img_draw, (int(width / 2), int(height / 2)), 3, color=(255, 255, 255),
                           thickness=-1)
            else:
                cv2.circle(img_draw, (int(height / 2), int(width / 2)), 3, color=(255, 255, 255),
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

            frame = Rotate(frame, int(leds_dic['cam_info'][cam_id]['display']['rotate']))
            img_draw = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img_gray = frame

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
                cv2.rectangle(img_draw, p1, p2, (255, 255, 255), 2, 1)
                IDX = bboxes[i]['idx']
                view_camera_infos(img_draw, ''.join([f'{IDX}']), int(newbox[0]), int(newbox[1]) - 10)

            # add graysum find center
            for i, newbox in enumerate(boxes):
                IDX = bboxes[i]['idx']
                ret, new_blobs = find_center(img_gray, IDX, int(newbox[0]), int(newbox[1]),
                                             int(newbox[2]), int(newbox[3]), blobs)
                if ret == DONE:
                    blobs = new_blobs

            KEY = cv2.waitKey(CAM_DELAY)

            # ToDo
            if ret == DONE:
                ret_status, min_blob = simple_solvePNP(cam_id, img_draw, blobs)
                if ret_status == SUCCESS:
                    leds_dic['cam_info'][cam_id]['blobs'] = min_blob
                    leds_dic['cam_info'][cam_id]['med_blobs'].append(min_blob)
                    leds_dic['cam_info'][cam_id]['detect_status'][1] += 1

                    if SENSOR_NAME == 'Rift':
                        cv2.rectangle(img_draw, (10, 10), (width - 10, height - 10), (255, 255, 255), 1)
                    else:
                        cv2.rectangle(img_draw, (10, 10), (height - 10, width - 10), (255, 255, 255), 1)
                    cam_ori = R.from_rotvec(leds_dic['cam_info'][cam_id]['RER']['C_R_T']['rvecs'].reshape(3)).as_quat()
                    cam_ori_euler = np.round_(get_euler_from_quat('xyz', cam_ori), 3)
                    cam_ori_quat = np.round_(get_quat_from_euler('xyz', cam_ori_euler), 8)
                    cam_pos = leds_dic['cam_info'][cam_id]['RER']['C_R_T']['tvecs'].reshape(3)
                    cv2.putText(img_draw, ''.join(['rot 'f'{cam_ori_euler}']),
                                (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), lineType=cv2.LINE_AA)
                    cv2.putText(img_draw, ''.join(['quat 'f'{cam_ori_quat}']),
                                (20, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), lineType=cv2.LINE_AA)
                    cv2.putText(img_draw, ''.join(['pos 'f'{cam_pos}']),
                                (20, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), lineType=cv2.LINE_AA)
                stacked = len(leds_dic['cam_info'][cam_id]['med_blobs'])
                cv2.putText(img_draw, ''.join([f'{stacked}', ' data stacked']),
                            (20, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), lineType=cv2.LINE_AA)

            # quit on ESC button
            if KEY & 0xFF == 27:  # Esc pressed
                cap.release()
                cv2.destroyAllWindows()
                break
            elif KEY == ord('e'):
                break
            elif KEY == ord('s'):
                if recording_start == 0:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    now = datetime.datetime.now()
                    recording_file_name = ''.join([f'{now}', '.avi'])
                    print('recording start', ' ', recording_file_name)
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    recording_out = cv2.VideoWriter(recording_file_name, fourcc, fps,
                                                    (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT))
                    recording_start = 1

                    leds_dic['cam_info'][cam_id]['track_cal']['recording'] = {'name': recording_file_name}

            elif KEY == ord('c') or leds_dic['cam_info'][cam_id]['detect_status'][1] >= LOOP_CNT:
                blob_array, rt_array = median_blobs(cam_id,
                                                    leds_dic['cam_info'][cam_id]['med_blobs'],
                                                    leds_dic['cam_info'][cam_id]['D_R_T_A'])

                leds_dic['cam_info'][cam_id]['med_blobs'] = blob_array
                leds_dic['cam_info'][cam_id]['D_R_T_A'] = rt_array

                # ToDo ????
                if recording_start == 1:
                    recording_out.release()
                    recording_start = 0

                print('capture done')

                break

            if recording_start == 1:
                recording_out.write(frame)

            # show frame
            cv2.imshow('MultiTracker', img_draw)

        # release camera frame
        if recording_start == 1:
            recording_out.release()

        cap.release()
        cv2.destroyAllWindows()

        bboxes.clear()
        blobs.clear()

    # remake 3D
    pairPoints('med_blobs', 'target_pts')
    coord2dto3d('target_pts', DYNAMIC_RT_MED)
    draw_ax_plot()




