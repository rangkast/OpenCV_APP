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
    leds_dic['cam_info'][0]['rotate'] = int(external_jdata['stereol']['rotate'])
    leds_dic['cam_info'][1]['rotate'] = int(external_jdata['stereor']['rotate'])

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

    if not cap1.isOpened() or not cap2.isOpened():
        sys.exit()

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
        imgL = Rotate(frame1, leds_dic['cam_info'][0]['rotate'])
        imgR = Rotate(frame2, leds_dic['cam_info'][1]['rotate'])
        view_camera_infos(imgL, f'{cap1_name}', 30, 35)
        cv2.circle(imgL, (int(height1 / 2), int(width1 / 2)), 3, color=(0, 0, 255),
                   thickness=-1)
        cv2.imshow('left camera', imgL)
        view_camera_infos(imgR, f'{cap2_name}', 30, 35)
        cv2.circle(imgR, (int(height2 / 2), int(width2 / 2)), 3, color=(0, 0, 255),
                   thickness=-1)
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

        print('width ', leds_dic['cam_info'][cam_id]['display']['width'])
        print('height ', leds_dic['cam_info'][cam_id]['display']['height'])

        # cap.set(cv2.CAP_PROP_FORMAT, cv2.CV_64FC1)

        while True:
            ret, frame = cap.read()
            if not ret:
                print('Cannot read video file')
                break
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = Rotate(frame, leds_dic['cam_info'][cam_id]['rotate'])
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
                    cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2, 1)
                    view_camera_infos(img_draw, ''.join([f'{IDX}']), int(x), int(y) - 10)
                    view_camera_infos(img_draw, ''.join(['[', f'{IDX}', '] '
                                                            , f' {x}'
                                                            , f' {y}']), 30, 35 + i * 30)

            view_camera_infos(img_draw, ''.join(['Cam[', f'{cam_id}', '] ', f'{video_src}']),
                              height - 500, 35)

            cv2.circle(img_draw, (int(height / 2), int(width / 2)), 3, color=(0, 0, 255),
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

            frame = Rotate(frame, leds_dic['cam_info'][cam_id]['rotate'])
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

                    cv2.rectangle(img_draw, (10, 10), (height - 10, width - 10), (255, 255, 255), 2)
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

                if recording_start == 1:
                    recording_out.release()
                    recording_start = 0
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




