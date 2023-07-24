from Advanced_Function import *

camera_matrix = [
    [np.array([[712.623, 0.0, 653.448],
               [0.0, 712.623, 475.572],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)],
]
default_dist_coeffs = np.zeros((4, 1))
default_cameraK = np.eye(3).astype(np.float64)
BLOB_SIZE = 150
TRACKER_PADDING = 3
CV_MAX_THRESHOLD = 255
CV_MIN_THRESHOLD = 150
MAX_LEVEL = 3
CAM_ID = 0
CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
UVC_MODE = 0
AUTO_LOOP = 0
undistort = 1

def sliding_window(data, window_size):
    for i in range(len(data) - window_size + 1):
        yield data[i:i + window_size]

def circular_sliding_window(data, window_size):
    data = list(data) + list(data[:window_size-1])  # Convert to list and append window_size-1 elements
    for i in range(len(data) - window_size + 1):
        yield data[i:i + window_size]

def auto_labeling():
    # Select the first camera device
    if UVC_MODE:
        cam_dev_list = terminal_cmd('v4l2-ctl', '--list-devices')
        camera_devices = init_model_json(cam_dev_list)
        print(camera_devices)
        camera_port = camera_devices[0]['port']
        # Open the video capture
        video = cv2.VideoCapture(camera_port)
        # Set the resolution
        video.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)
    else:
        image_files = sorted(glob.glob(os.path.join(script_dir, f"./render_img/rifts_right_9/test_1/" + '*.png')))
        frame_cnt = 0
        print('lenght of images: ', len(image_files))


    while video.isOpened() if UVC_MODE else True:
        print('\n')
        print(f"########## Frame {frame_cnt} ##########")
        if UVC_MODE:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_cnt)
            ret, frame_0 = video.read()
            if not ret:
                break
        else:
            # BLENDER와 확인해 보니 마지막 카메라 위치가 시작지점으로 돌아와서 추후 remake 3D 에서 이상치 발생 ( -1 )  
            if frame_cnt >= len(image_files):
                break
            frame_0 = cv2.imread(image_files[frame_cnt])
            filename = f"IMAGE Mode {os.path.basename(image_files[frame_cnt])}"
            if frame_0 is None or frame_0.size == 0:
                print(f"Failed to load {image_files[frame_cnt]}, frame_cnt:{frame_cnt}")
                continue
            frame_0 = cv2.resize(frame_0, (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT))



        draw_frame = frame_0.copy()
        _, frame_0 = cv2.threshold(cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                   cv2.THRESH_TOZERO)

        cv2.putText(draw_frame, f"frame_cnt {frame_cnt} [{filename}]", (draw_frame.shape[1] - 500, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        center_x, center_y = CAP_PROP_FRAME_WIDTH // 2, CAP_PROP_FRAME_HEIGHT // 2
        cv2.line(draw_frame, (0, center_y), (CAP_PROP_FRAME_WIDTH, center_y), (255, 255, 255), 1)
        cv2.line(draw_frame, (center_x, 0), (center_x, CAP_PROP_FRAME_HEIGHT), (255, 255, 255), 1) 


        # find Blob area by findContours
        blob_area = detect_led_lights(frame_0, TRACKER_PADDING, 5, 500)
        blobs = []

        for blob_id, bbox in enumerate(blob_area):
            (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            gcx, gcy, gsize = find_center(frame_0, (x, y, w, h))
            if gsize < BLOB_SIZE:
                continue 

            overlapping = check_blobs_with_pyramid(frame_0, draw_frame, x, y, w, h, MAX_LEVEL)
            if overlapping == True:
                cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 0, 255), 1, 1)
                cv2.putText(draw_frame, f"SEG", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                continue
            
            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
            blobs.append((gcx, gcy, bbox))

        blobs = sorted(blobs, key=lambda x:x[0]) ## 또는 l.sort(key=lambda x:x[1])
        CNT = len(blobs)
        # print('CNT ', CNT)
        if CNT >= 4:            
            window_size = 4
            def is_decreasing(arr):
                return all(x>y for x, y in zip(arr, arr[1:]))

            min_RER_global = float('inf')
            best_LED_NUMBER_global = None
            sliding_pos = None
            BRFS_GLOBAL = None
            RVEC_GLOBAL = None
            TVEC_GLOBAL = None
            for blob_idx in sliding_window(range(CNT), window_size):
                # print('blob_idx', blob_idx)
                
                candidates = []
                min_RER_blob = float('inf')
                best_LED_NUMBER_blob = None
                BRFS = None
                RVEC = None
                TVEC = None
                for i in blob_idx:
                    candidates.append((blobs[i][0], blobs[i][1]))
                    (x, y, w, h) = (int(blobs[i][2][0]), int(blobs[i][2][1]), int(blobs[i][2][2]), int(blobs[i][2][3]))
                    cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
                    # cv2.putText(draw_frame, f"{i}", (x, y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                points2D_D = np.array(np.array(candidates).reshape(len(candidates), -1), dtype=np.float64)
                points2D_U = np.array(cv2.undistortPoints(points2D_D, camera_matrix[0][0], camera_matrix[0][1])).reshape(-1, 2)
                # print('points2D_D\n', points2D_D)
                # print('points2D_U\n', points2D_U)
                for model_idx in circular_sliding_window(range(BLOB_CNT), window_size):
                    model_candidates = []
                    # print('model_idx ', model_idx)
                    LED_NUMBER = []
                    for j in model_idx:
                        # print(j)
                        LED_NUMBER.append(j)
                        model_candidates.append(MODEL_DATA[j])
                    points3D = np.array(model_candidates, dtype=np.float64)
                    # print('points3D\n', points3D)
                    points3D_reverse = np.flip(points3D, axis=0)
                    # print('points3D_reverse\n', points3D_reverse)
                    MODEL_BRFS = [points3D, points3D_reverse]
                    for brfs_idx, brfs in enumerate(MODEL_BRFS):
                        if brfs_idx:
                            LED_NUMBER = np.flip(LED_NUMBER)
                            
                        METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P

                        INPUT_ARRAY = [
                            CAM_ID,
                            brfs,
                            points2D_D if undistort == 0 else points2D_U,
                            camera_matrix[CAM_ID][0] if undistort == 0 else default_cameraK,
                            camera_matrix[CAM_ID][1] if undistort == 0 else default_dist_coeffs
                        ]
                        _, rvec, tvec, _ = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
                        # 로드리게즈 변환을 사용해 회전 벡터를 회전 행렬로 변환

                        # print('PnP_Solver rvec:', rvec, ' tvec:',  tvec)
                        # rvec이나 tvec가 None인 경우 continue
                        if rvec is None or tvec is None:
                            continue
                        
                        # rvec이나 tvec에 nan이 포함된 경우 continue
                        if np.isnan(rvec).any() or np.isnan(tvec).any():
                            continue

                        # rvec과 tvec의 모든 요소가 0인 경우 continue
                        if np.all(rvec == 0) and np.all(tvec == 0):
                            continue
                        # print('rvec:', rvec)
                        # print('tvec:', tvec)                        

                        RER = reprojection_error(brfs,
                                                 points2D_D,
                                                 rvec, tvec,
                                                 camera_matrix[CAM_ID][0],
                                                 camera_matrix[CAM_ID][1])
                        if RER < 1:
                            # print(f"{LED_NUMBER} RER {RER}")
                            # 순차적으로 감소하면서 최소 RER 값을 갖는 LED_NUMBER 업데이트
                            if RER < min_RER_blob and is_decreasing(LED_NUMBER):
                                min_RER_blob = RER
                                best_LED_NUMBER_blob = LED_NUMBER
                                BRFS = brfs
                                RVEC = rvec
                                TVEC = tvec
                

                # 각 슬라이딩 윈도우에서 최적의 결과를 전체 결과와 비교
                if min_RER_blob < min_RER_global:
                    min_RER_global = min_RER_blob
                    best_LED_NUMBER_global = best_LED_NUMBER_blob
                    sliding_pos = blob_idx
                    BRFS_GLOBAL = BRFS
                    RVEC_GLOBAL = RVEC
                    TVEC_GLOBAL = TVEC
            # projectPoints를 사용해 3D points를 2D image points로 변환
            image_points, _ = cv2.projectPoints(np.array(BRFS_GLOBAL),
                                                np.array(RVEC_GLOBAL),
                                                np.array(TVEC_GLOBAL),
                                                camera_matrix[CAM_ID][0],
                                                camera_matrix[CAM_ID][1])
            image_points = image_points.reshape(-1, 2)

            for point, led_num in zip(image_points, best_LED_NUMBER_global):
                pt = (int(point[0]), int(point[1]))
                # cv2.circle을 사용해 표시
                cv2.circle(draw_frame, tuple(pt), 2, (0, 0, 255), -1)
                # cv2.putText를 사용해 LED 번호 표시
                cv2.putText(draw_frame, f"{led_num}", (pt[0], pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            print(f"Best LED_NUMBER: {best_LED_NUMBER_global}, with RER: {min_RER_global} sliding_window: {sliding_pos}")
                           
        
        elif CNT == 3:
            window_size = 3
            for blob_idx in sliding_window(range(CNT), window_size):
                # print(blob_idx)
                candidates = []
                for i in blob_idx:
                    candidates.append((blobs[i][0], blobs[i][1]))
                    (x, y, w, h) = (int(blobs[i][2][0]), int(blobs[i][2][1]), int(blobs[i][2][2]), int(blobs[i][2][3]))
                    cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 0), 1, 1)
                    cv2.putText(draw_frame, f"{i}", (x, y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                points2D_D = np.array(np.array(candidates).reshape(len(candidates), -1), dtype=np.float64)
                points2D_U = np.array(cv2.undistortPoints(points2D_D, camera_matrix[0][0], camera_matrix[0][1])).reshape(-1, 2)
                # print('points2D_D\n', points2D_D)
                # print('points2D_U\n', points2D_U)
                for model_idx in sliding_window(range(BLOB_CNT), window_size):
                    model_candidates = []
                    # print(model_idx)
                    LED_NUMBER = []
                    for i in model_idx:
                        # print(i)
                        LED_NUMBER.append(i)
                        model_candidates.append(MODEL_DATA[i])
                    points3D = np.array(model_candidates, dtype=np.float64)
                    # print('points3D\n', points3D)
                    points3D_reverse = np.flip(points3D, axis=0)
                    # print('points3D_reverse\n', points3D_reverse)
                    MODEL_BRFS = [points3D, points3D_reverse]
                    for brfs_idx, brfs in enumerate(MODEL_BRFS):
                        X = np.array(brfs)
                        x = np.hstack((points2D_U, np.ones((points2D_U.shape[0], 1))))
                        # print('X ', X)
                        # print('x ', x)
                        poselib_result = poselib.p3p(x, X)
                        # print(X)
                        # print(poselib_result)
                        
                        if brfs_idx:
                            LED_NUMBER = np.flip(LED_NUMBER)
                                               
                        for solution_idx, pose in enumerate(poselib_result):
                            colors = [(255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
                            if is_valid_pose(pose):
                                quat = pose.q
                                tvec = pose.t
                                rotm = quat_to_rotm(quat)
                                rvec, _ = cv2.Rodrigues(rotm)
                                # print("PoseLib rvec: ", rvec.flatten(), ' tvec:', tvec)                               
                                image_points, _ = cv2.projectPoints(np.array(brfs),
                                    np.array(rvec),
                                    np.array(tvec),
                                    camera_matrix[CAM_ID][0],
                                    camera_matrix[CAM_ID][1])
                                image_points = image_points.reshape(-1, 2)
                                # print('image_points\n', image_points)
                                cam_pos, cam_dir, _ = calculate_camera_position_direction(rvec, tvec)
                                
                                ###############################            
                                visible_result = check_angle_and_facing(MODEL_DATA, DIRECTION, cam_pos, quat, LED_NUMBER)
                                # print(f"{brfs_idx} visible_result {visible_result}")
                                visible_status = SUCCESS
                                for blob_id, status in visible_result.items():
                                    if status == False:
                                        visible_status = ERROR
                                        # print(f"{solution_idx} pose unvisible led {blob_id}")
                                        break   
                                if visible_status == SUCCESS:
                                    print('candidates LED_NUMBER ', LED_NUMBER)
        if AUTO_LOOP:
            frame_cnt += 1
        # Display the resulting frame
        cv2.imshow('Frame', draw_frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('e'): 
            # Use 'e' key to exit the loop
            break
        elif key & 0xFF == ord('n'):  
            frame_cnt += 1     
        elif key & 0xFF == ord('b'):  
            frame_cnt -= 1    

    # Release everything when done
    if UVC_MODE == 1:
        video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(os.getcwd())

    MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/rifts_right_9.json"))
    BLOB_CNT = len(MODEL_DATA)

    print('PTS')
    for i, leds in enumerate(MODEL_DATA):
        print(f"{np.array2string(leds, separator=', ')},")

    auto_labeling()