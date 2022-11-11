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


class CameraWidget(QtWidgets.QWidget):
    def __init__(self, width, height, stream_dev_cam, aspect_ratio=False, parent=None, deque_size=1):
        try:
            super(CameraWidget, self).__init__(parent)
            # Initialize deque used to store frames read from the stream
            self.deque = deque(maxlen=deque_size)

            # So add offset to counter the padding
            self.offset = 16
            self.screen_width = width - self.offset
            self.screen_height = height - self.offset
            self.maintain_aspect_ratio = aspect_ratio

            self.camera_stream_link = stream_dev_cam

            # Flag to check if camera is valid/working
            self.online = False
            self.capture = None
            self.video_frame = QtWidgets.QLabel()

            self.load_dev_cam_stream()

            # Start background frame grabbing
            self.get_frame_thread = threading.Thread(target=self.get_frame, args=())
            self.get_frame_thread.daemon = True
            self.get_frame_thread.start()

            # Periodically set video frame to display
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.set_frame)
            self.timer.start(int(.5))
            print('Start camera: {}'.format(self.camera_stream_link))
        except:
            print('exception')
            traceback.print_exc()
        pass

    def load_dev_cam_stream(self):
        """Verifies stream link and open new stream if valid"""

        def load_dev_cam_stream_thread():
            if self.verify_camera_stream(self.camera_stream_link):
                self.capture = cv2.VideoCapture(self.camera_stream_link)
                self.online = True

        self.load_stream_thread = threading.Thread(target=load_dev_cam_stream_thread, args=())
        self.load_stream_thread.daemon = True
        self.load_stream_thread.start()

    def verify_camera_stream(self, link):

        """Attempts to receive a frame from given link"""
        cap = cv2.VideoCapture(link)
        if not cap.isOpened():
            return False
        cap.release()
        return True

    def get_frame(self):
        """Reads frame, resizes, and converts image to pixmap"""
        while True:
            try:
                if self.capture.isOpened() and self.online:
                    # Read next frame from stream and insert into deque
                    status, frame = self.capture.read()
                    if status:
                        cam_lock.acquire()
                        self.deque.append(frame)
                        cam_lock.release()
                    else:
                        self.capture.release()
                        self.online = False
                else:
                    # Attempt to reconnect
                    print('attempting to reconnect', self.camera_stream_link)
                    self.load_dev_cam_stream()
                    self.spin(2)
                self.spin(.001)
            except AttributeError:
                pass

    def spin(self, seconds):
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""

        time_end = time.time() + seconds
        while time.time() < time_end:
            QtWidgets.QApplication.processEvents()

    def set_frame(self):
        """Sets pixmap image to video frame"""

        if not self.online:
            self.spin(1)
            return

        if self.deque and self.online:
            cam_lock.acquire()
            # Grab latest frame
            frame = self.deque[-1]

            # Keep frame aspect ratio
            if self.maintain_aspect_ratio:
                self.frame = imutils.resize(frame, width=self.screen_width)
            # Force resize
            else:
                self.frame = cv2.resize(frame, (self.screen_width, self.screen_height))

            # Convert to pixmap and set to video frame
            self.img = QtGui.QImage(self.frame, self.frame.shape[1], self.frame.shape[0],
                                    QtGui.QImage.Format_RGB888).rgbSwapped()
            self.pix = QtGui.QPixmap.fromImage(self.img)
            self.video_frame.setPixmap(self.pix)
            cam_lock.release()

    def get_video_frame(self):
        return self.video_frame

    def mouseMoveEvent(self, e):
        x = e.x()
        y = e.y()

        text = 'x: {0}, y: {1}'.format(x, y)
        self.label.setText(text)
        self.label.adjustSize()


def exit_application():
    """Exit program event handler"""
    sys.exit(1)


def init_cam_lock():
    global cam_lock
    cam_lock = threading.Lock()


def single_tracker():
    for cam_id in range(len(leds_dic['cam_info'])):
        print('try to open:', leds_dic['cam_info'][cam_id]['port'])
        # 트랙커 객체 생성자 함수 리스트 ---①
        trackers = [cv2.legacy.TrackerBoosting_create,
                    cv2.legacy.TrackerMIL_create,
                    cv2.legacy.TrackerKCF_create,
                    cv2.legacy.TrackerTLD_create,
                    cv2.legacy.TrackerMedianFlow_create,
                    # cv2.legacy.TrackerGOTURN_create,  # 버그로 오류 발생
                    cv2.legacy.TrackerCSRT_create,
                    cv2.legacy.TrackerMOSSE_create]
        trackerIdx = 0  # 트랙커 생성자 함수 선택 인덱스
        tracker = None
        isFirst = True
        video_src = 0  # 비디오 파일과 카메라 선택 ---②
        video_src = leds_dic['cam_info'][cam_id]['port']

        cap = cv2.VideoCapture(video_src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FORMAT, cv2.CV_64FC1)

        win_name = 'Tracking APIs'
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Cannot read video file')
                break

            # Todo
            # frame, status, ret_blobs = check_frame_status(frame, cam_id, CV_THRESHOLD)8
            # adjust_boundary_filter(frame, cam_id)
            # status, new_blobs = find_blob_center(cam_id, frame, ret_blobs)
            # imshow_and_info(frame, cam_id, ret_blobs)

            img_draw = frame.copy()

            if tracker is None:  # 트랙커 생성 안된 경우
                cv2.putText(img_draw, "Press the Space to set ROI!!",
                            (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                ok, bbox = tracker.update(frame)  # 새로운 프레임에서 추적 위치 찾기 ---③
                (x, y, w, h) = bbox
                if ok:  # 추적 성공
                    cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)),
                                  (0, 255, 0), 2, 1)
                else:  # 추적 실패
                    cv2.putText(img_draw, "Tracking fail.", (100, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            trackerName = tracker.__class__.__name__
            cv2.putText(img_draw, str(trackerIdx) + ":" + trackerName, (100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(win_name, img_draw)
            key = cv2.waitKey(1) & 0xff
            # 스페이스 바 또는 비디오 파일 최초 실행 ---④
            if key == ord(' ') or (video_src != 0 and isFirst):
                isFirst = False
                roi = cv2.selectROI(win_name, frame, False)  # 초기 객체 위치 설정
                if roi[2] and roi[3]:  # 위치 설정 값 있는 경우
                    tracker = trackers[trackerIdx]()  # 트랙커 객체 생성 ---⑤
                    isInit = tracker.init(frame, roi)
            elif key in range(48, 56):  # 0~7 숫자 입력   ---⑥
                trackerIdx = key - 48  # 선택한 숫자로 트랙커 인덱스 수정
                if bbox is not None:
                    tracker = trackers[trackerIdx]()  # 선택한 숫자의 트랙커 객체 생성 ---⑦
                    isInit = tracker.init(frame, bbox)  # 이전 추적 위치로 추적 위치 초기화
            elif key == 27:
                break
        else:
            print("Could not open video")
        cap.release()
        cv2.destroyAllWindows()


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


def multi_tracker():
    leds_dic['test_status']['event_status'] = {'event': -1,
                                               'x': 0,
                                               'y': 0,
                                               'flag': 0,
                                               'prev_event': cv2.EVENT_LBUTTONUP}
    ## Select boxes
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
                    cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2, 1)
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
        capture_start = 0
        recording_start = 0
        prev_data_stack_num = -1
        # Process video and track objects
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            ret, img_contour_binary = cv2.threshold(frame, CV_FINDCONTOUR_LVL, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
            img_gray = cv2.cvtColor(img_contour_binary, cv2.COLOR_BGR2GRAY)

            # # Todo
            # cv2.imshow('img_gray', img_gray.shape[::-1])
            # print(img_gray.shape[::-1])
            # hL, wL = img_gray.shape[:2]
            #
            # print('hl ', hL, ' wl ', wL)
            # continue

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
                cv2.rectangle(img_gray, p1, p2, 255, 2, 1)
                IDX = bboxes[i]['idx']
                view_camera_infos(img_gray, ''.join([f'{IDX}']), int(newbox[0]), int(newbox[1]) - 10)

            KEY = cv2.waitKey(1)

            # key control
            if KEY == ord('c'):
                capture_start = 1

                # add graysum find center
                for i, newbox in enumerate(boxes):
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    IDX = bboxes[i]['idx']
                    ret, new_blobs = find_center(img_gray, IDX, int(newbox[0]), int(newbox[1]),
                                                 int(newbox[2]), int(newbox[3]), blobs)
                    if ret == DONE:
                        # print(new_blobs)
                        blobs = new_blobs

                # ToDo
                ret_status, min_blob = simple_solvePNP(cam_id, frame, blobs)
                # calibrate_camera(cam_id, img_gray, blobs)

                if ret_status == SUCCESS:
                    leds_dic['cam_info'][cam_id]['detect_status'][1] += 1
                    leds_dic['test_status']['cam_capture'] = DONE
                    leds_dic['cam_info'][cam_id]['blobs'] = min_blob
                    if RT_TEST == ENABLE:
                        leds_dic['cam_info'][cam_id]['detect_status'][2] += 1
                        # 여기에 data가 쌓임
                        #####
                        leds_dic['cam_info'][cam_id]['med_blobs'].append(min_blob)
                        #####

                cv2.rectangle(img_gray, (10, 10), (CAP_PROP_FRAME_WIDTH - 10, CAP_PROP_FRAME_HEIGHT - 10), 255, 2, 2)
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
            elif KEY & 0xFF == 27:  # Esc pressed
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

                    leds_dic['cam_info'][cam_id]['track_cal']['recording'].append(
                        {'name': recording_file_name})

            elif KEY & 0xFF in range(48, 58):  # 0~9 숫자 입력   ---⑥
                NUMBER = KEY - 48
                if prev_data_stack_num != NUMBER and capture_start == 1:
                    blob_array, rt_array = median_blobs(cam_id,
                                                        leds_dic['cam_info'][cam_id]['med_blobs'],
                                                        leds_dic['cam_info'][cam_id]['D_R_T_A'])

                    prev_data_stack_num = NUMBER
                    capture_start = 0

                    leds_dic['cam_info'][cam_id]['track_cal']['data'].append(
                        {'idx': NUMBER, 'blobs': blob_array, 'R_T': rt_array})

                    leds_dic['cam_info'][cam_id]['med_blobs'].clear()
                    leds_dic['cam_info'][cam_id]['D_R_T_A'].clear()

                    print(leds_dic['cam_info'][cam_id]['track_cal'])

                if recording_start == 1:
                    recording_out.release()
                    recording_start = 0

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

        # remake 3D
        pair_xy(cam_id, 'target_pts')
        refactor_3d_point(cam_id, 'target_pts')
        draw_ax_plot()

        if len(leds_dic['cam_info'][cam_id]['track_cal']['data']) == 2:
            print('store data')
            json_file = ''.join(['jsons/test_result/', f'{JSON_FILE}'])
            group_data = OrderedDict()
            group_data['stereol'] = {'cam_id': cam_id,
                                     'blobs': leds_dic['cam_info'][cam_id]['track_cal']['data'][0]['blobs'],
                                     'rvecs': leds_dic['cam_info'][cam_id]['track_cal']['data'][0]['R_T']['rvecs'][:, 0].tolist(),
                                     'tvecs': leds_dic['cam_info'][cam_id]['track_cal']['data'][0]['R_T']['tvecs'][:, 0].tolist(),
                                     'file': leds_dic['cam_info'][cam_id]['track_cal']['recording'][0]['name']}
            group_data['stereor'] = {'cam_id': cam_id,
                                     'blobs': leds_dic['cam_info'][cam_id]['track_cal']['data'][1]['blobs'],
                                     'rvecs': leds_dic['cam_info'][cam_id]['track_cal']['data'][1]['R_T']['rvecs'][:, 0].tolist(),
                                     'tvecs': leds_dic['cam_info'][cam_id]['track_cal']['data'][1]['R_T']['tvecs'][:, 0].tolist(),
                                     'file': leds_dic['cam_info'][cam_id]['track_cal']['recording'][1]['name']}
            rw_json_data(WRITE, json_file, group_data)


def refactor_3d_point(cam_id, target):
    print('start coord refine')
    for i in range(len(leds_dic[target])):
        # if len(leds_data[target][i]['pair_xy']) > 0:
        #     print('led_num ', i, ' ', leds_data[target][i]['pair_xy'])

        led_pair_cnt = len(leds_dic[target][i]['pair_xy'])
        if led_pair_cnt < 2:
            print(f'Error LED Num {i} has no more than 2-cameras')
            leds_dic[target][i]['remake_3d'] = 'error'
        else:
            comb_led = list(itertools.combinations(leds_dic[target][i]['pair_xy'], 2))
            for data in comb_led:
                print('comb_led idx: ', i, ' : ', data)
                result = coordRefactor(leds_dic['cam_info'][cam_id], data[0], data[1])
                leds_dic[target][i]['remake_3d'].append(
                    {'idx': i, 'cam_l': data[0]['cidx'], 'cam_r': data[1]['cidx'], 'coord': result})

    if DEBUG == ENABLE:
        for i in range(len(leds_dic[target])):
            if leds_dic[target][i]['remake_3d'] != 'error':
                print(leds_dic[target][i]['remake_3d'])


def coordRefactor(cam_info, cam_l, cam_r):
    cam_l_id = cam_l['cidx']
    cam_r_id = cam_r['cidx']

    l_rvec = cam_info['track_cal']['data'][cam_l_id]['R_T']['rvecs']
    r_rvec = cam_info['track_cal']['data'][cam_r_id]['R_T']['rvecs']
    l_tvec = cam_info['track_cal']['data'][cam_l_id]['R_T']['tvecs']
    r_tvec = cam_info['track_cal']['data'][cam_r_id]['R_T']['tvecs']

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
                                          (cam_l['cx'], cam_l['cy']),
                                          (cam_r['cx'], cam_r['cy']))
    homog_points = triangulation.transpose()

    get_points = cv2.convertPointsFromHomogeneous(homog_points)

    return get_points


def pair_xy(cam_id, target):
    print('start pair_xy')
    for track_data in leds_dic['cam_info'][cam_id]['track_cal']['data']:
        idx_num = int(track_data['idx'])
        for leds in track_data['blobs']:
            led_num = int(leds['idx'])
            cx = leds['cx']
            cy = leds['cy']
            leds_dic[target][led_num]['pair_xy'].append({'cidx': idx_num,
                                                         'led_num': led_num,
                                                         'cx': cx,
                                                         'cy': cy})
    if DEBUG == ENABLE:
        for i in range(len(leds_dic[target])):
            if len(leds_dic[target][i]['pair_xy']) > 0:
                print(leds_dic[target][i]['pair_xy'])


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


def check_frame_status(frame, cam_id, threshold):
    new_frame, blobs = rt_contour_detect(frame, cam_id, threshold)
    status, ret_blobs = translate_led_id(new_frame, cam_id, blobs)
    return new_frame, status, ret_blobs


def imshow_and_info(frame, cam_id, blobs):
    bgr_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    view_camera_infos(bgr_image, cam_id, leds_dic['cam_info'][cam_id]['port'],
                      CAP_PROP_FRAME_WIDTH - 100)
    print_frame_data(bgr_image, cam_id, blobs)

    # draw_area_spec(bgr_image)
    cv2.imshow(SINGLE_CAMERA_TITLE, bgr_image)
    cv2.setMouseCallback(SINGLE_CAMERA_TITLE, onMouse)


def onMouse(e, x_coord, y_coord, flags, param):
    leds_dic['test_status']['event_status']['event'] = e
    leds_dic['test_status']['event_status']['x'] = x_coord
    leds_dic['test_status']['event_status']['y'] = y_coord
    leds_dic['test_status']['event_status']['flag'] = flags


def draw_area_spec(frame):
    global start_x, start_y
    if leds_dic['test_status']['event_status']['event'] != -1:
        # print('e:', event, ' x:', x, ' y:', y)
        x = leds_dic['test_status']['event_status']['x']
        y = leds_dic['test_status']['event_status']['y']
        if leds_dic['test_status']['event_status']['flag'] & cv2.EVENT_FLAG_SHIFTKEY:
            if leds_dic['test_status']['event_status']['prev_event'] == cv2.EVENT_LBUTTONDOWN and \
                    leds_dic['test_status']['event_status']['event'] == cv2.EVENT_LBUTTONUP:
                print('drag up', ' x:', x, ' y:', y)
                start_x = -1
                start_y = -1
            elif leds_dic['test_status']['event_status']['prev_event'] == cv2.EVENT_LBUTTONUP and \
                    leds_dic['test_status']['event_status']['event'] == cv2.EVENT_LBUTTONDOWN:
                print('drag down', ' x:', x, ' y:', y)
                start_x = x
                start_y = y

            elif leds_dic['test_status']['event_status']['prev_event'] == cv2.EVENT_LBUTTONDOWN and \
                    leds_dic['test_status']['event_status']['event'] == cv2.EVENT_MOUSEMOVE:
                if start_x != -1 and start_y != -1:
                    cx = int((start_x + x) / 2)
                    cy = int((start_y + y) / 2)
                    dist = np.sqrt(np.power(start_x - x, 2) + np.power(start_y - y, 2))
                    if PRINT_FRAME_INFOS == ENABLE:
                        cv2.circle(frame, (cx, cy), int(dist), color=(255, 255, 0), thickness=1)

            if leds_dic['test_status']['event_status']['event'] == cv2.EVENT_LBUTTONDOWN or \
                    leds_dic['test_status']['event_status']['event'] == cv2.EVENT_LBUTTONUP:
                leds_dic['test_status']['event_status']['prev_event'] = leds_dic['test_status']['event_status']['event']


def exit_application():
    """Exit program event handler"""
    sys.exit(1)


def init_cam_lock():
    global cam_lock
    cam_lock = threading.Lock()
