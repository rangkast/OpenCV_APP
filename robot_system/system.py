from robot_system.common import *
from robot_system.calibration import *

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
TAG = '[SYSTEM]'


# Set Camera led blob spec area
# Check json file and matching blobs
def robot_camera_setting():
    print(TAG, robot_camera_setting.__name__)
    camera_cnt = 0
    camera_mode = ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_camera_mode()
    while True:
        # two camera connect At the same time
        if camera_mode == CAMERA_MODE.DUAL_CAMERA:
            cam_l_id = camera_cnt % 2
            cam_r_id = (camera_cnt + 1) % 2
        elif camera_mode == CAMERA_MODE.MULTI_CAMERA:
            cam_l_id = camera_cnt
            cam_r_id = camera_cnt + 1

        dev_name = ROBOT_SYSTEM_DATA[CAM_INFO][cam_l_id].get_dev()
        json_file = ''.join(['jsons/' f'{CAM_INFO}/', f'{dev_name}/', BLOB_AREA_FILE])
        json_data = rw_json_data(READ, json_file, None)
        imgL_name = ROBOT_SYSTEM_DATA[CAM_INFO][cam_l_id].get_name()
        imgR_name = ROBOT_SYSTEM_DATA[CAM_INFO][cam_r_id].get_name()
        bboxes = get_spec_data_from_json(json_data, imgL_name, imgR_name, camera_cnt, camera_cnt + 1)

        cam_l_port = ROBOT_SYSTEM_DATA[CAM_INFO][cam_l_id].get_port()
        cam_L = cv2.VideoCapture(cam_l_port)
        set_display_setting(cam_L, cam_l_id)
        cam_r_port = ROBOT_SYSTEM_DATA[CAM_INFO][cam_r_id].get_port()
        cam_R = cv2.VideoCapture(cam_r_port)
        set_display_setting(cam_R, cam_r_id)

        if not cam_L.isOpened() or not cam_R.isOpened():
            sys.exit()

        W, H, _ = ROBOT_SYSTEM_DATA[CAM_INFO][cam_l_id].get_display_info()

        while True:
            ret1, frame1 = cam_L.read()
            ret2, frame2 = cam_R.read()
            if not ret1 or not ret2:
                break

            # see frames
            stacked_frame = np.hstack((rotate(frame1, cam_l_id), rotate(frame2, cam_r_id)))
            draw_frame = stacked_frame.copy()

            if DEBUG > DEBUG_LEVEL.DISABLE:
                if len(bboxes) > 0:
                    l_cnt = 0
                    r_cnt = 0
                    for i, data in enumerate(bboxes):
                        (x, y, w, h) = data['bbox']
                        IDX = data['idx']
                        side = data['side']
                        cv2.rectangle(draw_frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 1, 1)
                        view_camera_infos(draw_frame, ''.join([f'{IDX}']), int(x), int(y) - 10)
                        if side == LEFT:
                            view_camera_infos(draw_frame, ''.join(['[', f'{IDX}', ']'
                                                                      , f' {x}'
                                                                      , f' {y}']), 30, 70 + l_cnt * 30)
                            l_cnt += 1
                        if side == RIGHT:
                            x = x - W
                            view_camera_infos(draw_frame, ''.join(['[', f'{IDX}', ']'
                                                                      , f' {x}'
                                                                      , f' {y}']), W + 30, 70 + r_cnt * 30)
                            r_cnt += 1

            KEY = cv2.waitKey(CAM_DELAY)

            if KEY & 0xFF == 27:
                print(TAG, 'ESC pressed')
                cam_L.release()
                cam_R.release()
                cv2.destroyAllWindows()
                return ERROR
            elif KEY == ord('e'):
                cam_L.release()
                cam_R.release()
                cv2.destroyAllWindows()
                return SUCCESS
            elif KEY == ord('c'):
                print(TAG, 'clear area')
                bboxes.clear()
            elif KEY == ord('a'):
                # set graysum area by bbox
                cv2.imshow("L/R frame", draw_frame)
                bbox = cv2.selectROI("L/R frame", draw_frame)

                while True:
                    inputs = input('input led number: ')
                    if inputs.isdigit():
                        input_number = int(inputs)
                        if input_number in range(0, 25):
                            print(TAG, 'led num ', input_number)
                            print(TAG, 'bbox ', bbox)
                            (x, y, w, h) = bbox
                            side = LEFT
                            if x >= W:
                                side = RIGHT
                            bboxes.append({'idx': input_number, 'bbox': bbox, 'side': side})
                            break
                        else:
                            print(TAG, 'led range over')
                    else:
                        if inputs == 'exit':
                            bboxes.clear()
                            break
            elif KEY == ord('s'):
                if len(bboxes) > 0:
                    print(TAG, 'try to update')
                    left_data = []
                    right_data = []
                    for i, data in enumerate(bboxes):
                        side = data['side']
                        if side == LEFT:
                            left_data.append(data)
                        else:
                            right_data.append(data)

                    lr_position = L_CONTROLLER if ROBOT_SYSTEM_DATA[
                                                      SYSTEM_SETTING].get_lr_position() == LR_POSITION.LEFT else R_CONTROLLER

                    if json_data != ERROR and json_data is not None:
                        print(TAG, 'exist')
                        print(json_data)

                        # ToDo
                        if str(camera_cnt) in json_data:
                            detect = 0
                            for i, data_i in enumerate(json_data[str(camera_cnt)]):
                                if imgL_name in data_i:
                                    detect = 1
                                    json_data[str(camera_cnt)][i][imgL_name][lr_position] = {'spec': left_data}
                            if detect == 0:
                                json_data[str(camera_cnt)].append({imgL_name: {lr_position: {'spec': left_data}}})
                        else:
                            json_data[camera_cnt] = [{imgL_name: {lr_position: {'spec': left_data}}}]

                        if str(camera_cnt + 1) in json_data:
                            detect = 0
                            for i, data_i in enumerate(json_data[str(camera_cnt + 1)]):
                                if imgR_name in data_i:
                                    detect = 1
                                    json_data[str(camera_cnt + 1)][i][imgR_name][lr_position] = {'spec': right_data}
                            if detect == 0:
                                json_data[str(camera_cnt + 1)].append({imgR_name: {lr_position: {'spec': right_data}}})
                        else:
                            json_data[camera_cnt + 1] = [{imgR_name: {lr_position: {'spec': right_data}}}]

                    else:
                        print(TAG, 'not exist')
                        json_data = OrderedDict()
                        json_data[camera_cnt] = [{ROBOT_SYSTEM_DATA[CAM_INFO][camera_cnt].get_name(): {
                            lr_position: {'spec': left_data}}}]
                        json_data[camera_cnt + 1] = [{ROBOT_SYSTEM_DATA[CAM_INFO][camera_cnt + 1].get_name(): {
                            lr_position: {'spec': right_data}}}]

                    # Write json data
                    rw_json_data(WRITE, json_file, json_data)

                else:
                    print(TAG, 'area not selected')
                break
            elif KEY == ord('n'):
                if DEBUG > DEBUG_LEVEL.LV_1:
                    print(TAG, 'go next camera')
                break

            # DrawFrame
            draw_common_ui(draw_frame, CAMERA_SETTING_MODE, cam_l_id, cam_r_id, camera_cnt)
            cv2.imshow("L/R frame", draw_frame)

        # End of stereo cam
        cam_L.release()
        cam_R.release()
        cv2.destroyAllWindows()

        camera_cnt += 2

        if camera_cnt >= len(ROBOT_SYSTEM_DATA[CAM_INFO]) and camera_mode != CAMERA_MODE.DUAL_CAMERA:
            if DEBUG > DEBUG_LEVEL.LV_1:
                print(TAG, 'All camera looped')
            break

    return SUCCESS


# 1.Gathering blobs center (graysum)
# 2.Median Blobs and R_T (IQR)
# 3.PairPoints
def robot_camera_default():
    print(TAG, robot_camera_default.__name__)
    camera_cnt = 0
    group_cnt = 0
    camera_mode = ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_camera_mode()
    while True:
        # two camera connect At the same time
        if camera_mode == CAMERA_MODE.DUAL_CAMERA:
            cam_l_id = camera_cnt % 2
            cam_r_id = (camera_cnt + 1) % 2
        elif camera_mode == CAMERA_MODE.MULTI_CAMERA:
            cam_l_id = camera_cnt
            cam_r_id = camera_cnt + 1

        dev_name = ROBOT_SYSTEM_DATA[CAM_INFO][cam_l_id].get_dev()
        json_file = ''.join(['jsons/' f'{CAM_INFO}/', f'{dev_name}/', BLOB_AREA_FILE])
        json_data = rw_json_data(READ, json_file, None)
        imgL_name = ROBOT_SYSTEM_DATA[CAM_INFO][cam_l_id].get_name()
        imgR_name = ROBOT_SYSTEM_DATA[CAM_INFO][cam_r_id].get_name()
        bboxes = get_spec_data_from_json(json_data, imgL_name, imgR_name, camera_cnt, camera_cnt + 1)

        cam_l_port = ROBOT_SYSTEM_DATA[CAM_INFO][cam_l_id].get_port()
        cam_L = cv2.VideoCapture(cam_l_port)
        set_display_setting(cam_L, cam_l_id)
        cam_r_port = ROBOT_SYSTEM_DATA[CAM_INFO][cam_r_id].get_port()
        cam_R = cv2.VideoCapture(cam_r_port)
        set_display_setting(cam_R, cam_r_id)

        if len(bboxes) <= 0:
            print(TAG, 'SPEC IS NULL')
            cam_L.release()
            cam_R.release()
            cv2.destroyAllWindows()
            return SUCCESS

        if not cam_L.isOpened() or not cam_R.isOpened():
            sys.exit()

        W, H, _ = ROBOT_SYSTEM_DATA[CAM_INFO][cam_l_id].get_display_info()

        LOOP_FUNCTION(make_measurement_data,
                      cam_l=[camera_cnt],
                      cam_r=[camera_cnt + 1])

        while True:
            ret1, frame1 = cam_L.read()
            ret2, frame2 = cam_R.read()
            if not ret1 or not ret2:
                break

            # Draw frames
            stacked_frame = np.hstack((rotate(frame1, cam_l_id), rotate(frame2, cam_r_id)))
            draw_frame = stacked_frame.copy()
            ret, img_filtered = cv2.threshold(stacked_frame, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
            img_gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)

            KEY = cv2.waitKey(CAM_DELAY)

            if KEY & 0xFF == 27:
                print(TAG, 'ESC pressed')
                cam_L.release()
                cam_R.release()
                cv2.destroyAllWindows()
                return ERROR
            elif KEY == ord('n'):
                print(TAG, 'go next camera')
                break

            # get blob center by using graysum
            # L/R image attached and get center simultaneously
            if len(bboxes) > 0:
                for i, data in enumerate(bboxes):
                    (x, y, w, h) = data['bbox']
                    IDX = data['idx']
                    side = data['side']
                    cam_id = camera_cnt if side == LEFT else camera_cnt + 1
                    if DEBUG > DEBUG_LEVEL.DISABLE:
                        cv2.rectangle(draw_frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 1, 1)
                        view_camera_infos(draw_frame, ''.join([f'{IDX}']), int(x), int(y) - 10)

                    ret, new_curr_blobs = find_center(img_gray, IDX, side, W,
                                                      (x, y, w, h), cam_id)
                    if ret == SUCCESS:
                        ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id].set_blob(new_curr_blobs)

            # led blob format check
            LOOP_FUNCTION(generate_led_blobs_format,
                          cam_l=[camera_cnt],
                          cam_r=[camera_cnt + 1])

            # Get R_T Vectors
            LOOP_FUNCTION(camera_pose_estimation,
                          cam_l=[camera_cnt],
                          cam_r=[camera_cnt + 1])

            # DrawFrame
            draw_common_ui(draw_frame,
                           CALIBRATION_MODE if ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_mode() == MODE.CALIBRATION_MODE
                           else SYSTEM_SETTING_MODE, cam_l_id, cam_r_id, camera_cnt)

            cv2.imshow("L/R frame", draw_frame)

        # Median Blobs and R_T
        LOOP_FUNCTION(median_blobs,
                      cam_l=[camera_cnt],
                      cam_r=[camera_cnt + 1])

        # Data changed if CALIBRATION_MODE SET
        # DATA set if SYSTEM_SETTING_MODE SET
        LOOP_FUNCTION(static_rt_func,
                      cam_l=[camera_cnt],
                      cam_r=[camera_cnt + 1])

        # PairPoints
        LOOP_FUNCTION(pair_points,
                      cam_l=[camera_cnt, group_cnt],
                      cam_r=[camera_cnt + 1, group_cnt])

        # End of stereo cam
        cam_L.release()
        cam_R.release()
        cv2.destroyAllWindows()

        camera_cnt += 2
        group_cnt += 1

        if camera_cnt >= len(ROBOT_SYSTEM_DATA[CAM_INFO]) and camera_mode != CAMERA_MODE.DUAL_CAMERA:
            print(TAG, 'All camera looped')

            # make remake 3D
            robot_data_remake_3d()

            break

    return SUCCESS


def robot_animate_tracker():
    print(TAG, robot_animate_tracker.__name__)
    ANIMATION()

    return SUCCESS


class ANIMATION:
    def __init__(self, **kw):
        plt.style.use('default')
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        cam_l_port = ROBOT_SYSTEM_DATA[CAM_INFO][0].get_port()
        cam_L = cv2.VideoCapture(cam_l_port)
        cam_r_port = ROBOT_SYSTEM_DATA[CAM_INFO][1].get_port()
        cam_R = cv2.VideoCapture(cam_r_port)

        self.ani = FuncAnimation(plt.gcf(), self.animate_func, fargs=(ax, cam_L, cam_R))
        # draw origin data
        draw_dots(ax, [D3D, 'blue'])

        ax.scatter(0, 0, 0, marker='o', color='k', s=20)
        ax.set_xlim([-0.2, 0.2])
        ax.set_xlabel('X')
        ax.set_ylim([-0.2, 0.2])
        ax.set_ylabel('Y')
        ax.set_zlim([-0.2, 0.2])
        ax.set_zlabel('Z')
        scale = 1.5
        f = zoom_factory(ax, base_scale=scale)
        plt.tight_layout()
        plt.show()

    def animate_func(self, *args):
        # Function Call
        axs = args[1]
        cam_l_img = args[2]
        cam_r_img = args[3]

        c_cnt = 0
        g_cnt = 0
        caml_id = 0
        camr_id = 1

        set_display_setting(cam_l_img, caml_id)
        set_display_setting(cam_r_img, camr_id)

        dev_name = ROBOT_SYSTEM_DATA[CAM_INFO][caml_id].get_dev()
        json_file = ''.join(['jsons/' f'{CAM_INFO}/', f'{dev_name}/', BLOB_AREA_FILE])
        json_data = rw_json_data(READ, json_file, None)
        imgL_name = ROBOT_SYSTEM_DATA[CAM_INFO][caml_id].get_name()
        imgR_name = ROBOT_SYSTEM_DATA[CAM_INFO][camr_id].get_name()
        # ToDo
        bboxes = get_spec_data_from_json(json_data, imgL_name, imgR_name, c_cnt, c_cnt + 1)
        if len(bboxes) <= 0:
            print(TAG, 'SPEC IS NULL')
            cam_l_img.release()
            cam_r_img.release()
            cv2.destroyAllWindows()
            self.ani.event_source.stop()
            return SUCCESS

        if not cam_l_img.isOpened() or not cam_r_img.isOpened():
            print(TAG, 'LR_img not opened')
            self.ani.event_source.stop()
            return ERROR

        W, H, _ = ROBOT_SYSTEM_DATA[CAM_INFO][caml_id].get_display_info()

        LOOP_FUNCTION(make_measurement_data,
                      cam_l=[c_cnt],
                      cam_r=[c_cnt + 1])

        # Specify the tracker type
        trackerType = "CSRT"
        # Create MultiTracker object
        multiTracker = cv2.legacy.MultiTracker_create()
        tracker_start = STOP
        capture_start = STOP
        while True:
            ret1, frame1 = cam_l_img.read()
            ret2, frame2 = cam_r_img.read()
            if not ret1 or not ret2:
                break

            # Draw frames
            stacked_frame = np.hstack((rotate(frame1, caml_id), rotate(frame2, camr_id)))
            draw_frame = stacked_frame.copy()
            ret, img_filtered = cv2.threshold(stacked_frame, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
            img_gray = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)

            # Initialize MultiTracker
            if tracker_start == STOP:
                for i, data in enumerate(bboxes):
                    multiTracker.add(createTrackerByName(trackerType), img_gray, data['bbox'])
            tracker_start = START
            # get updated location of objects in subsequent frames
            qq, new_boxes = multiTracker.update(img_gray)

            # get blob center by using graysum
            # L/R image attached and get center simultaneously
            if len(bboxes) > 0:
                for i, data in enumerate(bboxes):
                    (x, y, w, h) = (
                        int(new_boxes[i][0]), int(new_boxes[i][1]), int(new_boxes[i][2]), int(new_boxes[i][3]))

                    IDX = data['idx']
                    side = data['side']
                    cam_id = c_cnt if side == LEFT else c_cnt + 1
                    if DEBUG > DEBUG_LEVEL.DISABLE:
                        cv2.rectangle(draw_frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 255, 255), 1, 1)
                        view_camera_infos(draw_frame, ''.join([f'{IDX}']), int(x), int(y) - 10)

                    if capture_start == START:
                        ret, new_curr_blobs = find_center(img_gray, IDX, side, W,
                                                          (x, y, w, h), cam_id)
                        if ret == SUCCESS:
                            ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id].set_blob(new_curr_blobs)

            KEY = cv2.waitKey(CAM_DELAY)

            if KEY & 0xFF == 27:
                print(TAG, 'ESC pressed')
                cam_l_img.release()
                cam_r_img.release()
                cv2.destroyAllWindows()
                return ERROR
            elif KEY == ord('e'):
                # End of stereo cam
                cam_l_img.release()
                cam_r_img.release()
                cv2.destroyAllWindows()
                print(TAG, 'event stop')
                robot_module_lsm()
                # draw remake 3d
                draw_dots(axs, [D3DR, 'red'])
                return SUCCESS

            elif KEY == ord('s'):
                capture_start = START
            elif KEY == ord('c'):
                print(TAG, 'capture')

                # led blob format check
                LOOP_FUNCTION(generate_led_blobs_format,
                              cam_l=[c_cnt],
                              cam_r=[c_cnt + 1])

                # Get R_T Vectors
                LOOP_FUNCTION(camera_pose_estimation,
                              cam_l=[c_cnt],
                              cam_r=[c_cnt + 1])

                # Median Blobs and R_T
                LOOP_FUNCTION(median_blobs,
                              cam_l=[c_cnt],
                              cam_r=[c_cnt + 1])

                # Data changed if CALIBRATION_MODE SET
                # DATA set if SYSTEM_SETTING_MODE SET
                LOOP_FUNCTION(static_rt_func,
                              cam_l=[c_cnt],
                              cam_r=[c_cnt + 1])

                # PairPoints
                LOOP_FUNCTION(pair_points,
                              cam_l=[c_cnt, g_cnt],
                              cam_r=[c_cnt + 1, g_cnt])

                # end of loop
                c_cnt += 2
                g_cnt += 1

                # make next measurement data
                LOOP_FUNCTION(make_measurement_data,
                              cam_l=[c_cnt],
                              cam_r=[c_cnt + 1])
                capture_start = STOP

                # remake 3d_data
                robot_data_remake_3d()
                # draw remake 3d
                draw_dots(axs, [D3DR, 'green'])
                plt.draw()

            # DrawFrame
            draw_common_ui(draw_frame,
                           CALIBRATION_MODE if ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_mode() == MODE.CALIBRATION_MODE
                           else SYSTEM_SETTING_MODE, caml_id, camr_id, c_cnt)
            cv2.imshow("L/R frame", draw_frame)

        return SUCCESS


def robot_data_remake_3d():
    print(TAG, robot_data_remake_3d.__name__)

    # remake 3d point by cv2.triangulatePoints
    refactor_3d_point()

    return SUCCESS


# Least Squared Method
# remake 3D points changed data by LSM
def robot_lsm():
    print(TAG, robot_lsm.__name__)
    pts = ROBOT_SYSTEM_DATA[LED_INFO]

    origin_pts = []
    before_pts = []

    # get led number only remake DONE
    for i in range(len(pts)):
        if len(pts[i].get_remake_3d()) > 0:
            before_pts.append(
                [pts[i].get_remake_3d()[0].blob.x, pts[i].get_remake_3d()[0].blob.y, pts[i].get_remake_3d()[0].blob.z])
            origin_pts.append(pts[i].get_pos())

    A = np.vstack([np.array(before_pts).T, np.ones(len(before_pts))]).T
    Rt = np.linalg.lstsq(A, np.array(origin_pts), rcond=None)[0]
    Pd_est = np.matmul(Rt.T, A.T)
    Err = np.array(origin_pts) - Pd_est.T

    # remake_3d -> np.array
    # camera id stable
    for i in range(len(pts)):
        if len(pts[i].get_remake_3d()) > 0:
            new_remake_3d = REMAKE_3D(pts[i].get_remake_3d()[0].cam_l,
                                      pts[i].get_remake_3d()[0].cam_r,
                                      BLOB_3D(pts[i].get_idx(), Pd_est.T[i][0], Pd_est.T[i][1], Pd_est.T[i][2]))
            pts[i].set_remake_3d(new_remake_3d)

    if DEBUG > DEBUG_LEVEL.DISABLE:
        print(TAG, 'remake_3d data changed by LSM')
        print(TAG, "Rt=\n", Rt.T)
        print(TAG, "Pd_est=\n", Pd_est)
        print(TAG, "Err=\n", Err)
        print(TAG, 'origin_pts ', origin_pts)
        print(TAG, 'before_pts ', before_pts)
        print(TAG, 'after_pts ')
        for i in range(len(pts)):
            if len(pts[i].get_remake_3d()) > 0:
                print([pts[i].get_remake_3d()[0].blob])

    return SUCCESS


# remake 3D points changed data by rigid transform 3D
def robot_module_lsm():
    print(TAG, robot_module_lsm.__name__)
    # X,Y,Z coordination
    axis_cnt = 3

    for key, value in ROBOT_SYSTEM_DATA[GROUP_DATA_INFO].items():
        origin_pts = []
        before_pts = []
        led_array = []
        cam_l = -1
        cam_r = -1

        # get led number only remake DONE
        for g_data in value:
            if len(g_data.get_pair_xy()) > 0:
                remake_blob = g_data.get_remake_3d().blob
                cam_l = g_data.get_remake_3d().cam_l
                cam_r = g_data.get_remake_3d().cam_r
                led_num = int(remake_blob.idx)
                led_array.append(led_num)
                origin_pts.append(ROBOT_SYSTEM_DATA[LED_INFO][led_num].get_pos())
                before_pts.append([remake_blob.x, remake_blob.y, remake_blob.z])

        origin_pts = np.array(origin_pts)
        before_pts = np.array(before_pts)
        led_blob_cnt = len(led_array)

        # make 3xN matrix
        A = np.array([[0 for j in range(led_blob_cnt)] for i in range(axis_cnt)], dtype=float)
        B = np.array([[0 for j in range(led_blob_cnt)] for i in range(axis_cnt)], dtype=float)
        for r in range(led_blob_cnt):
            for c in range(axis_cnt):
                B[c][r] = origin_pts[r][c]
                A[c][r] = before_pts[r][c]

        # calculation rigid_transform
        ret_R, ret_t = rigid_transform_3D(A, B)
        C = (ret_R @ A) + ret_t

        for r, led_idx in enumerate(led_array):
            NEW_BLOB_3D = BLOB_3D()
            NEW_BLOB_3D.idx = led_idx
            NEW_BLOB_3D.x = float(C[0][r])
            NEW_BLOB_3D.y = float(C[1][r])
            NEW_BLOB_3D.z = float(C[2][r])

            # Remake_3d data change
            ROBOT_SYSTEM_DATA[GROUP_DATA_INFO][key][led_idx].set_remake_3d(REMAKE_3D(cam_l, cam_r, NEW_BLOB_3D))

        diff = np.array(C - B)
        err = C - B
        dist = []
        for i in range(len(diff[0])):
            dist.append(np.sqrt(np.power(diff[0][i], 2) + np.power(diff[1][i], 2) + np.power(diff[2][i], 2)))
        err = err * err
        err = np.sum(err)
        rmse = np.sqrt(err / len(diff[0]))

        if DEBUG > DEBUG_LEVEL.LV_2:
            print(TAG, key)
            print(TAG, value)
            print(TAG, 'rmse')
            print(rmse)
            print(TAG, 'A')
            print(A)
            print(TAG, 'B')
            print(B)
            print(TAG, 'C')
            print(C)
            print(TAG, 'diff')
            print(diff)
            print(TAG, 'dist')
            print(dist)

    return SUCCESS


def robot_print_result():
    print(TAG, robot_print_result.__name__)
    show_3d_plot()
    show_stem_plot()
    plt.show()
    return SUCCESS


def robot_dump_data():
    print(TAG, robot_dump_data.__name__)
    formattedDate = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_file_name = ''.join(['dump_', f'{formattedDate}', '.pickle'])
    file = ''.join(['jsons/dump/', dump_file_name])
    data = OrderedDict()
    data[LED_INFO] = ROBOT_SYSTEM_DATA[LED_INFO]
    data[CAM_INFO] = ROBOT_SYSTEM_DATA[CAM_INFO]
    data[SYSTEM_SETTING] = ROBOT_SYSTEM_DATA[SYSTEM_SETTING]
    data[GROUP_DATA_INFO] = ROBOT_SYSTEM_DATA[GROUP_DATA_INFO]
    data[MEASUREMENT_INFO] = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO]

    pickle_data(WRITE, file, data)

    return SUCCESS


def robot_init_data_arrays():
    print(TAG, robot_init_data_arrays.__name__, 'start')
    # Init Camera device info
    ROBOT_SYSTEM_DATA[CAM_INFO] = init_data_array(terminal_cmd('v4l2-ctl', '--list-devices'))
    # Init LED Blobs info
    ROBOT_SYSTEM_DATA[LED_INFO] = init_coord_json(ORIGIN)
    # Init Measurement data
    ROBOT_SYSTEM_DATA[MEASUREMENT_INFO] = []
    # Init Group Dictionary
    ROBOT_SYSTEM_DATA[GROUP_DATA_INFO] = {}
