from Advanced_Function import *

# MODEL_DATA, DIRECTION = init_coord_json(f"{script_dir}/jsons/specs/semi_slam_curve.json")
# TARGET_DEVICE = 'SEMI_SLAM_CURVE'
MODEL_DATA, DIRECTION = init_coord_json(f"{script_dir}/jsons/specs/semi_slam_polyhedron.json")
TARGET_DEVICE = 'SEMI_SLAM_POLYHEDRON'
MODEL_DATA = np.array(MODEL_DATA)
BLOB_CNT = len(MODEL_DATA)
CV_MAX_THRESHOLD = 255
CV_MIN_THRESHOLD = 150
BLOB_SIZE = 50
TRACKER_PADDING = 5
LOOP_CNT = 1
VIDEO_MODE = 1


BLOB_INFO = pickle_data(READ, 'BLOB_INFO.pickle', None)['BLOB_INFO']
CAMERA_INFO = pickle_data(READ, 'CAMERA_INFO.pickle', None)['CAMERA_INFO']    
for key, camera_info in CAMERA_INFO.items():
    print('frame_cnt: ', key)
    print('ORVEC:', camera_info['OPENCV']['rt']['rvec'].flatten())
    print('OTVEC:', camera_info['OPENCV']['rt']['tvec'].flatten())
    LED_NUMBER = camera_info['LED_NUMBER']
    points3D = camera_info['points3D']
    points2D = camera_info['points2D']['greysum']
    points2D_U = camera_info['points2D_U']['greysum']
    print('LED_NUMBER: ', LED_NUMBER)
    print('points2D_U: ',points2D_U)
    print('points3D\n',points3D)


CAMERA_M = [
    # WMTD306N100AXM
    [np.array([[712.623, 0.0, 653.448],
               [0.0, 712.623, 475.572],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)],
    # WMTD305L6003D6
    [np.array([[716.896, 0.0, 668.902],
               [0.0, 716.896, 460.618],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.07542], [-0.026874], [0.006662], [-0.000775]], dtype=np.float64)],
]

def get_blob_area(frame):
    filtered_blob_area = []
    blob_area = detect_led_lights(frame, TRACKER_PADDING, 5, 500)
    for _, bbox in enumerate(blob_area):
        (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        gcx,gcy, gsize = find_center(frame, (x, y, w, h))
        if gsize < BLOB_SIZE:
            continue
        filtered_blob_area.append((gcx, gcy, (x, y, w, h)))   

    return filtered_blob_area

def get_coords(frame, bboxes, CAM_ID):
    points2D_D = []
    points2D_U = []
    points3D = []
    LED_NUMBERS = []
    for box in bboxes:
        (x, y, w, h) = (box['bbox'])
        gcx, gcy, gsize = find_center(frame, (x, y, w, h))
        if gsize < BLOB_SIZE:
            continue
        points2D_D.append([gcx, gcy])
        LED_NUMBERS.append(int(box['idx']))

    points3D = MODEL_DATA[list(LED_NUMBERS), :]
    points2D_D = np.array(np.array(points2D_D).reshape(len(points2D_D), -1), dtype=np.float64)
    points2D_U = cv2.undistortPoints(points2D_D, CAMERA_M[CAM_ID][0], CAMERA_M[CAM_ID][1])
    points2D_U = np.array(np.array(points2D_U).reshape(len(points2D_U), -1), dtype=np.float64)

    return points3D, points2D_D, points2D_U, LED_NUMBERS

def calc_pose(frame, bboxes, CAM_ID):
    ret_status = SUCCESS    
    points3D, points2D_D, points2D_U, LED_NUMBERS = get_coords(frame, bboxes, CAM_ID)
    length = len(points2D_D)
    if length >= 5:
        METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC
    elif length == 4:
        METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P
    else:
        ret_status = ERROR

    if ret_status == SUCCESS:
        INPUT_ARRAY = [
            CAM_ID,
            points3D,
            points2D_U,
            default_cameraK,
            default_dist_coeffs
        ]
        _, rvec, tvec, _ = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
        # rvec이나 tvec가 None인 경우 continue
        if rvec is None or tvec is None:
            ret_status = ERROR
        
        # rvec이나 tvec에 nan이 포함된 경우 continue
        if np.isnan(rvec).any() or np.isnan(tvec).any():
            ret_status = ERROR

        # rvec과 tvec의 모든 요소가 0인 경우 continue
        if np.all(rvec == 0) and np.all(tvec == 0):
            ret_status = ERROR
        # print('rvec:', rvec)
        # print('tvec:', tvec)
    
    if ret_status == ERROR:
        return ERROR, points2D_U, ERROR, ERROR
    
    rvec_reshape = np.array(rvec).reshape(-1, 1)
    tvec_reshape = np.array(tvec).reshape(-1, 1)

    return [rvec_reshape, tvec_reshape], points2D_U, points3D, LED_NUMBERS


def stereo_blob_setting(camera_devices):
    if VIDEO_MODE == 1:
        cam_L = cv2.VideoCapture(camera_devices[0]['port'])
        cam_L.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
        cam_L.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)

        cam_R = cv2.VideoCapture(camera_devices[1]['port'])
        cam_R.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
        cam_R.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)

    left_bboxes = []    
    json_data = rw_json_data(READ, f"{script_dir}/render_img/stereo/blob_area_left.json", None)
    if json_data != ERROR:
        left_bboxes = json_data['bboxes']

    right_bboxes = []    
    json_data = rw_json_data(READ, f"{script_dir}/render_img/stereo/blob_area_right.json", None)
    if json_data != ERROR:
        right_bboxes = json_data['bboxes']

    while True:
        if VIDEO_MODE == 1:
            ret1, frame1 = cam_L.read()
            ret2, frame2 = cam_R.read()
            if not ret1 or not ret2:
                break
        else:
            frame1 = cv2.imread( f"{script_dir}/render_img/stereo/frame_0026.png")
            if frame1 is None or frame1.size == 0:
                continue
            frame1 = cv2.resize(frame1, (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT))
            frame2 = cv2.imread( f"{script_dir}/render_img/stereo/frame_0044.png")
            if frame2 is None or frame2.size == 0:
                continue
            frame2 = cv2.resize(frame2, (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT))

        draw_frame1 = frame1.copy()
        draw_frame2 = frame2.copy()
        
        _, frame1 = cv2.threshold(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                cv2.THRESH_TOZERO)
        _, frame2 = cv2.threshold(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                            cv2.THRESH_TOZERO)        
        
        cv2.namedWindow('LEFT CAMERA')
        filtered_blob_area_left = get_blob_area(frame1)
        partial_click_event = functools.partial(click_event, frame=frame1, blob_area_0=filtered_blob_area_left, bboxes=left_bboxes)
        cv2.setMouseCallback('LEFT CAMERA', partial_click_event)
        draw_blobs_and_ids(draw_frame1, filtered_blob_area_left, left_bboxes)

        cv2.namedWindow('RIGHT CAMERA')
        filtered_blob_area_right = get_blob_area(frame2)
        partial_click_event = functools.partial(click_event, frame=frame2, blob_area_0=filtered_blob_area_right, bboxes=right_bboxes)
        cv2.setMouseCallback('RIGHT CAMERA', partial_click_event)
        draw_blobs_and_ids(draw_frame2, filtered_blob_area_right, right_bboxes)

        if len(left_bboxes) >= 4 and len(right_bboxes) >= 4:
            LRTVEC, L_points2D_U, L_points3D, LED_NUMBERS = calc_pose(frame1, left_bboxes, 0)
            if LRTVEC != ERROR:
                print('LEFT')
                print(LRTVEC[0].flatten())
                print(LRTVEC[1].flatten())
                print(LED_NUMBERS)
                print('points2D_U: ',L_points2D_U)
                print(L_points3D)
            image_points, _ = cv2.projectPoints(L_points3D,
                                                np.array(LRTVEC[0]),
                                                np.array(LRTVEC[1]),
                                                CAMERA_M[0][0],
                                                CAMERA_M[0][1])
            image_points = image_points.reshape(-1, 2)
            ###################################
            for i, point in enumerate(image_points):
                pt = (int(point[0]), int(point[1]))
                cv2.circle(draw_frame1, pt, 5, (255, 255, 0), -1)
                cv2.putText(draw_frame1, str(LED_NUMBERS[i]), (pt[0], pt[1]- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)

            RRTVEC, R_points2D_U, R_points3D, LED_NUMBERS = calc_pose(frame2, right_bboxes, 1)
            if RRTVEC != ERROR:
                print('RIGHT')
                print(RRTVEC[0].flatten())
                print(RRTVEC[1].flatten())
                print(LED_NUMBERS)
                print('points2D_U: ',R_points2D_U)
                print(R_points3D)
            image_points, _ = cv2.projectPoints(R_points3D,
                                                np.array(RRTVEC[0]),
                                                np.array(RRTVEC[1]),
                                                CAMERA_M[1][0],
                                                CAMERA_M[1][1])
            image_points = image_points.reshape(-1, 2)
            ###################################
            for i, point in enumerate(image_points):
                pt = (int(point[0]), int(point[1]))
                cv2.circle(draw_frame2, pt, 5, (255, 255, 0), -1)
                cv2.putText(draw_frame2, str(LED_NUMBERS[i]), (pt[0], pt[1]- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)

        cv2.imshow('LEFT CAMERA', draw_frame1)
        cv2.imshow('RIGHT CAMERA', draw_frame2)

        KEY = cv2.waitKey(1)
        if KEY & 0xFF == 27:
            break
        elif KEY == ord('c'):
            print('clear area')
            left_bboxes.clear()
            right_bboxes.clear()

        elif KEY == ord('s'):
            print('save blob area')
            json_data = OrderedDict()
            json_data['bboxes'] = left_bboxes
            json_data['LRTVEC'] = np.array(LRTVEC).tolist()
            rw_json_data(WRITE, f"{script_dir}/render_img/stereo/blob_area_left.json", json_data)

            json_data = OrderedDict()
            json_data['bboxes'] = right_bboxes
            json_data['RRTVEC'] = np.array(RRTVEC).tolist()
            rw_json_data(WRITE, f"{script_dir}/render_img/stereo/blob_area_right.json", json_data)
    if VIDEO_MODE == 1:
        cam_L.release()
        cam_R.release()
    cv2.destroyAllWindows()


def calibration(camera_devices):
    if VIDEO_MODE == 1:
        cam_L = cv2.VideoCapture(camera_devices[0]['port'])
        cam_L.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
        cam_L.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)

        cam_R = cv2.VideoCapture(camera_devices[1]['port'])
        cam_R.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
        cam_R.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)

    LRTVEC = NOT_SET
    RRTVEC = NOT_SET
    left_bboxes = []    
    json_data = rw_json_data(READ, f"{script_dir}/render_img/stereo/blob_area_left.json", None)
    if json_data != ERROR:
        left_bboxes = json_data['bboxes']
        LRTVEC = json_data['LRTVEC']
    print('LRTVEC')
    print(LRTVEC)

    right_bboxes = []    
    json_data = rw_json_data(READ, f"{script_dir}/render_img/stereo/blob_area_right.json", None)
    if json_data != ERROR:
        right_bboxes = json_data['bboxes']
        RRTVEC = json_data['RRTVEC']
    print('RRTVEC')
    print(RRTVEC)

    ba_3d_dict = {}
    frame_cnt = 0
    while True:
        if VIDEO_MODE == 1:
            ret1, frame1 = cam_L.read()
            ret2, frame2 = cam_R.read()
            if not ret1 or not ret2:
                break
        else:
            frame1 = cv2.imread( f"{script_dir}/render_img/stereo/frame_0026.png")
            if frame1 is None or frame1.size == 0:
                continue
            frame1 = cv2.resize(frame1, (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT))
            frame2 = cv2.imread( f"{script_dir}/render_img/stereo/frame_0044.png")
            if frame2 is None or frame2.size == 0:
                continue
            frame2 = cv2.resize(frame2, (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT))
        
        if frame_cnt > LOOP_CNT:
            break

        draw_frame1 = frame1.copy()
        draw_frame2 = frame2.copy()
        
        _, frame1 = cv2.threshold(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
        _, frame2 = cv2.threshold(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)        
        
        _, L_points2D_D, L_points2D_U, LED_NUMBERS = get_coords(frame1, left_bboxes, 0)
        _, R_points2D_D, R_points2D_U, LED_NUMBERS = get_coords(frame2, right_bboxes, 1)

        for pts in L_points2D_D:
            cv2.circle(draw_frame1, (int(pts[0]), int(pts[1])), 3, (0,255,0), -1)
        cv2.putText(draw_frame1, f"{frame_cnt}", (draw_frame1.shape[1] - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for pts in R_points2D_D:
            cv2.circle(draw_frame2, (int(pts[0]), int(pts[1])), 3, (0,255,0), -1)  
        cv2.putText(draw_frame2, f"{frame_cnt}", (draw_frame2.shape[1] - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        REMAKE_3D = remake_3d_point(default_cameraK, default_cameraK,
                                    {'rvec': np.array(LRTVEC[0]).reshape(-1, 1), 'tvec': np.array(LRTVEC[1]).reshape(-1, 1)},
                                    {'rvec': np.array(RRTVEC[0]).reshape(-1, 1), 'tvec': np.array(RRTVEC[1]).reshape(-1, 1)},
                                    L_points2D_U, R_points2D_U).reshape(-1, 3)
        # REMAKE_3D = remake_3d_point(default_cameraK, default_cameraK,
        #                     {'rvec': CAMERA_INFO['25']['OPENCV']['rt']['rvec'], 'tvec': CAMERA_INFO['25']['OPENCV']['rt']['tvec']},
        #                     {'rvec': CAMERA_INFO['43']['OPENCV']['rt']['rvec'], 'tvec': CAMERA_INFO['43']['OPENCV']['rt']['tvec']},
        #                     CAMERA_INFO['25']['points2D_U']['greysum'], CAMERA_INFO['43']['points2D_U']['greysum']).reshape(-1, 3)  
        print('REMAKE_3D')
        print(REMAKE_3D)
        print('LED_NUMBERS')
        print(LED_NUMBERS)

        for i, led_num in enumerate(LED_NUMBERS):
            if led_num not in ba_3d_dict:
                    ba_3d_dict[led_num] = []
            ba_3d_dict[led_num].append(REMAKE_3D[i])

        frame_cnt += 1
        cv2.imshow('LEFT CAMERA', draw_frame1)
        cv2.imshow('RIGHT CAMERA', draw_frame2)

        KEY = cv2.waitKey(1)
        if KEY & 0xFF == 27:
            break

    if VIDEO_MODE == 1:
        cam_L.release()
        cam_R.release()

    cv2.destroyAllWindows()

    # DO LSM
    print('#################### IQR  ####################')
    IQR_ARRAY = []
    TARGET_DATA = []
    for blob_id, points_3d in ba_3d_dict.items():
        TARGET_DATA.append(MODEL_DATA[int(blob_id)])
        acc_blobs = points_3d.copy()
        acc_blobs_length = len(acc_blobs)
        if acc_blobs_length == 0:
            print('acc_blobs_length is 0 ERROR')
            continue

        remove_index_array = []
        med_blobs = [[], [], []]
        for blobs in acc_blobs:
            med_blobs[0].append(blobs[0])
            med_blobs[1].append(blobs[1])
            med_blobs[2].append(blobs[2])
            
        detect_outliers(med_blobs, remove_index_array)

        count = 0
        for index in remove_index_array:
            med_blobs[0].pop(index - count)
            med_blobs[1].pop(index - count)
            med_blobs[2].pop(index - count)
            count += 1

        mean_med_x = round(np.mean(med_blobs[0]), 8)
        mean_med_y = round(np.mean(med_blobs[1]), 8)
        mean_med_z = round(np.mean(med_blobs[2]), 8)
        IQR_ARRAY.append([mean_med_x, mean_med_y, mean_med_z])
        print(f"mean_med of IQR for blob_id {blob_id}: [{mean_med_x} {mean_med_y} {mean_med_z}]")       

    TARGET_DATA = np.array(TARGET_DATA)
    if TARGET_DEVICE ==  'SEMI_SLAM_PLANE':
        IQR_ARRAY_LSM = module_lsm_2D(TARGET_DATA, IQR_ARRAY)
    else:
        IQR_ARRAY_LSM = module_lsm_3D(TARGET_DATA, IQR_ARRAY)
    IQR_ARRAY_LSM = [[round(x, 8) for x in sublist] for sublist in IQR_ARRAY_LSM]

    from Advanced_Calibration import init_plot
    ax1, ax2 = init_plot(MODEL_DATA)

    print('IQR_ARRAY_LSM')
    for blob_id, points_3d in enumerate(IQR_ARRAY_LSM):
        print(f"{points_3d},")
        ax1.scatter(points_3d[0], points_3d[1], points_3d[2], color='red', alpha=0.3, marker='o', s=7)

    cam_pos, cam_dir, _ = calculate_camera_position_direction(np.array(LRTVEC[0]), np.array(LRTVEC[1]))
    ax1.scatter(*cam_pos, c='red', marker='o', label=f"LEFT POS")
    ax1.quiver(*cam_pos, *cam_dir, color='red', label=f"LEFT DIR", length=0.1)    

    cam_pos, cam_dir, _ = calculate_camera_position_direction(np.array(RRTVEC[0]), np.array(RRTVEC[1]))
    ax1.scatter(*cam_pos, c='blue', marker='o', label=f"RIGHT POS")
    ax1.quiver(*cam_pos, *cam_dir, color='blue', label=f"RIGHT DIR", length=0.1)    

    plt.show()

if __name__ == "__main__":    
    cam_dev_list = terminal_cmd('v4l2-ctl', '--list-devices')
    camera_devices = init_model_json(cam_dev_list)
    print(camera_devices)

    print('PTS')
    for i, leds in enumerate(MODEL_DATA):
        print(f"{np.array2string(leds, separator=', ')},")
    print('DIR')
    for i, dir in enumerate(DIRECTION):
        print(f"{np.array2string(dir, separator=', ')},")


    # show_calibrate_data(np.array(MODEL_DATA), np.array(DIRECTION))
    # stereo_blob_setting(camera_devices)

    calibration(camera_devices)



