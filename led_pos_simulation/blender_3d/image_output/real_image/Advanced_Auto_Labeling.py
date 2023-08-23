from Advanced_Function import *
from data_class import *
import Advanced_Cython_Functions

RER_SPEC = 3.0
BLOB_SIZE = 30
TRACKER_PADDING = 3
CV_MAX_THRESHOLD = 255
CV_MIN_THRESHOLD = 150
MAX_LEVEL = 3
CAM_ID = 0
CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
UVC_MODE = 1
AUTO_LOOP = 0
undistort = 1



'''
Solutions
1 : sliding window 순서대로 할당
2 : sliding window x purmutations
3 : translation Matrix x projectPoints
4 : object tracking
'''
SOLUTION = 3

def sliding_window(data, window_size):
    for i in range(len(data) - window_size + 1):
        yield data[i:i + window_size]

def circular_sliding_window(data, window_size):
    data = list(data) + list(data[:window_size-1])  # Convert to list and append window_size-1 elements
    for i in range(len(data) - window_size + 1):
        yield data[i:i + window_size]

def TtoA(T):
    A = []
    for i, t in enumerate(T):
        for tt in t:
            A.append(np.array(tt))
    return np.array(A)


def AtoT(A):
    return torch.Tensor(np.array(A))

if SOLUTION == 1:
    def auto_labeling():
        frame_cnt = 0
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
            image_files = sorted(glob.glob(os.path.join(script_dir, f"./tmp/render/ARCTURAS/rotation_60/" + '*.png'))) 

            # image_files = sorted(glob.glob(os.path.join(script_dir, f"./render_img/rifts_right_9/test_1/" + '*.png')))        
            print('lenght of images: ', len(image_files))

        # Initialize each blob ID with a copy of the structure
        for blob_id in range(BLOB_CNT):
            BLOB_INFO[blob_id] = copy.deepcopy(BLOB_INFO_STRUCTURE)

        while video.isOpened() if UVC_MODE else True:      
            if UVC_MODE:
                ret, frame_0 = video.read()
                filename = f"VIDEO Mode {camera_port}"
                if not ret:
                    break
            else:
                print(f"########## Frame {frame_cnt} ##########")
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

            # cv2.namedWindow('Frame')
            # partial_click_event = functools.partial(click_event, frame=frame_0)
            # cv2.setMouseCallback('Frame', partial_click_event)


            # find Blob area by findContours
            blob_area = detect_led_lights(frame_0, TRACKER_PADDING)
            blobs = []

            for blob_id, bbox in enumerate(blob_area):
                (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                gcx, gcy, gsize = find_center(frame_0, (x, y, w, h))
                if gsize < BLOB_SIZE:
                    continue 
                # overlapping = check_blobs_with_pyramid(frame_0, draw_frame, x, y, w, h, MAX_LEVEL)
                # if overlapping == True:
                #     cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 0, 255), 1, 1)
                #     cv2.putText(draw_frame, f"SEG", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                #     continue
                
                cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
                blobs.append((gcx, gcy, bbox))

            blobs = sorted(blobs, key=lambda x:x[0]) ## 또는 l.sort(key=lambda x:x[1])
            CNT = len(blobs)
            # print('CNT ', CNT)
            if CNT >= 4:            
                window_size = 6
                min_RER_global = float('inf')
                best_LED_NUMBER_global = None
                sliding_pos = None
                BRFS_GLOBAL = None
                RVEC_GLOBAL = None
                TVEC_GLOBAL = None
                POINTS2D_U_GLOBAL = None
                POINTS2D_D_GLOBAL = None
                for blob_idx in sliding_window(range(CNT), window_size):
                    # print('blob_idx', blob_idx)
                    
                    candidates = []
                    min_RER_blob = float('inf')
                    best_LED_NUMBER_blob = None
                    BRFS = None
                    RVEC = None
                    TVEC = None
                    points2D_D_local = None
                    points2D_U_local = None
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
                                
                            METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC

                            INPUT_ARRAY = [
                                CAM_ID,
                                brfs,
                                points2D_D if undistort == 0 else points2D_U,
                                camera_matrix[CAM_ID][0] if undistort == 0 else default_cameraK,
                                camera_matrix[CAM_ID][1] if undistort == 0 else default_dist_coeffs
                            ]
                            _, rvec, tvec, _ = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)

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
                            if RER < RER_SPEC:
                                # pos, dir = world_location_rotation_from_opencv(rvec, tvec.flatten())
                                # ax.quiver(*pos, *dir, color='g', label='Direction Right', length=0.1)                                              
                                # cam_pos, cam_dir, _ = calculate_camera_position_direction(rvec, tvec)
                                # observed_rot = R.from_rotvec(rvec.reshape(3)).as_quat()
                                # quaternion = (observed_rot[0], observed_rot[1], observed_rot[2], observed_rot[3])
                                # print(f"CNT {CNT} {LED_NUMBER} RER {RER} blob_idx {blob_idx} cam_pos {cam_pos}")
                                # visible_result = check_angle_and_facing(MODEL_DATA, DIRECTION, cam_pos, quaternion, LED_NUMBER)
                                # print(f"1. visible_result {visible_result}")
                                # visible_result = check_simple_facing(MODEL_DATA, cam_pos, LED_NUMBER, angle_spec=80)
                                # print(f"2. visible_result {visible_result}")
                                plt.draw()
                                if RER < min_RER_blob:
                                    min_RER_blob = RER
                                    best_LED_NUMBER_blob = LED_NUMBER
                                    BRFS = brfs
                                    RVEC = rvec
                                    TVEC = tvec
                                    points2D_D_local = points2D_D
                                    points2D_U_local = points2D_U
                    

                    # 각 슬라이딩 윈도우에서 최적의 결과를 전체 결과와 비교
                    if min_RER_blob < min_RER_global:
                        min_RER_global = min_RER_blob
                        best_LED_NUMBER_global = best_LED_NUMBER_blob
                        sliding_pos = blob_idx
                        BRFS_GLOBAL = BRFS
                        RVEC_GLOBAL = RVEC
                        TVEC_GLOBAL = TVEC
                        POINTS2D_D_GLOBAL = points2D_D_local
                        POINTS2D_U_GLOBAL = points2D_U_local
                
                # print('BRFS_GLOBAL\n', BRFS_GLOBAL)
                # print('RVEC_GLOBAL\n', RVEC_GLOBAL)
                # print('TVEC_GLOBAL\n', TVEC_GLOBAL)
                if BRFS_GLOBAL is not None and RVEC_GLOBAL is not None and TVEC_GLOBAL is not None:
                    # print('RVEC: ', RVEC_GLOBAL.flatten())
                    # print('TVEC: ', TVEC_GLOBAL.flatten())
                    rvec_reshape = np.array(RVEC_GLOBAL).reshape(-1, 1)
                    tvec_reshape = np.array(TVEC_GLOBAL).reshape(-1, 1)
                    CAMERA_INFO[f"{frame_cnt}"] = copy.deepcopy(CAMERA_INFO_STRUCTURE)
                    CAMERA_INFO[f"{frame_cnt}"]['points3D'] = BRFS_GLOBAL
                    CAMERA_INFO[f"{frame_cnt}"]['points2D']['greysum'] = POINTS2D_D_GLOBAL
                    CAMERA_INFO[f"{frame_cnt}"]['points2D_U']['greysum'] = POINTS2D_U_GLOBAL            
                    CAMERA_INFO[f"{frame_cnt}"]['LED_NUMBER'] = best_LED_NUMBER_global                
                    CAMERA_INFO[f"{frame_cnt}"]['OPENCV']['rt']['rvec'] = rvec_reshape
                    CAMERA_INFO[f"{frame_cnt}"]['OPENCV']['rt']['tvec'] = tvec_reshape

                    for bid, blob_id in enumerate(best_LED_NUMBER_global):
                            BLOB_INFO[blob_id]['points2D_D']['greysum'].append(POINTS2D_D_GLOBAL[bid])
                            BLOB_INFO[blob_id]['points2D_U']['greysum'].append(POINTS2D_U_GLOBAL[bid])
                            BLOB_INFO[blob_id]['OPENCV']['rt']['rvec'].append(rvec_reshape)
                            BLOB_INFO[blob_id]['OPENCV']['rt']['tvec'].append(tvec_reshape)
                            BLOB_INFO[blob_id]['OPENCV']['status'].append(DONE)

                    rvec_euler = np.round_(get_euler_from_quat('xyz', R.from_rotvec(RVEC_GLOBAL.reshape(3)).as_quat()), 3)
                    cv2.putText(draw_frame, f"Rot {rvec_euler}", (draw_frame.shape[1] - 500, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    # projectPoints를 사용해 3D points를 2D image points로 변환
                    image_points, _ = cv2.projectPoints(np.array(BRFS_GLOBAL),
                                                        np.array(RVEC_GLOBAL),
                                                        np.array(TVEC_GLOBAL),
                                                        camera_matrix[CAM_ID][0],
                                                        camera_matrix[CAM_ID][1])
                    image_points = image_points.reshape(-1, 2)

                    for point, led_num in zip(image_points, best_LED_NUMBER_global):
                        pt = (int(point[0]), int(point[1]))

                        cv2.circle(draw_frame, tuple(pt), 2, (0, 0, 255), -1)
                        cv2.putText(draw_frame, f"{led_num}", (pt[0], pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # print(f"Best LED_NUMBER: {best_LED_NUMBER_global}, with RER: {min_RER_global} sliding_window: {sliding_pos}")
                    if AUTO_LOOP and UVC_MODE:
                        frame_cnt += 1
                                
            
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
                                    quaternion = pose.q
                                    tvec = pose.t
                                    rotm = quat_to_rotm(quaternion)
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
                                    visible_result = check_angle_and_facing(MODEL_DATA, DIRECTION, cam_pos, quaternion, LED_NUMBER)
                                    # print(f"{brfs_idx} visible_result {visible_result}")
                                    visible_status = SUCCESS
                                    for blob_id, status in visible_result.items():
                                        if status == False:
                                            visible_status = ERROR
                                            # print(f"{solution_idx} pose unvisible led {blob_id}")
                                            break   
                                    if visible_status == SUCCESS:
                                        print('candidates LED_NUMBER ', LED_NUMBER)
            if AUTO_LOOP and UVC_MODE == 0:
                frame_cnt += 1
            # Display the resulting frame
            cv2.imshow('Frame', draw_frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('e'): 
                # Use 'e' key to exit the loop
                break
            elif key & 0xFF == ord('n'):
                if AUTO_LOOP == 0 and UVC_MODE == 0:
                    frame_cnt += 1     
            elif key & 0xFF == ord('b'):
                if AUTO_LOOP== 0 and UVC_MODE == 0:
                    frame_cnt -= 1    

        data = OrderedDict()
        data['BLOB_INFO'] = BLOB_INFO
        pickle_data(WRITE, 'BLOB_INFO.pickle', data)
        data = OrderedDict()
        data['CAMERA_INFO'] = CAMERA_INFO
        pickle_data(WRITE, 'CAMERA_INFO.pickle', data)

        # Release everything when done
        if UVC_MODE == 1:
            video.release()
        cv2.destroyAllWindows()
elif SOLUTION == 2:
    def auto_labeling():
        frame_cnt = 0
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
            print('lenght of images: ', len(image_files))


        # Initialize each blob ID with a copy of the structure
        for blob_id in range(BLOB_CNT):
            BLOB_INFO[blob_id] = copy.deepcopy(BLOB_INFO_STRUCTURE)

        while video.isOpened() if UVC_MODE else True:
            print('\n')        
            if UVC_MODE:
                ret, frame_0 = video.read()
                filename = f"VIDEO Mode {camera_port}"
                if not ret:
                    break
            else:
                print(f"########## Frame {frame_cnt} ##########")
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
            blob_area = detect_led_lights(frame_0, TRACKER_PADDING)
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
            print('blobs ', blobs)
            CNT = len(blobs)
            # print('CNT ', CNT)
            if CNT >= 4:            
                window_size = 4
                for blob_idx in sliding_window(range(CNT), window_size):
                    print('blob_idx', blob_idx)                
                    candidates = []
                    for i in blob_idx:
                        candidates.append((blobs[i][0], blobs[i][1]))
                        (x, y, w, h) = (int(blobs[i][2][0]), int(blobs[i][2][1]), int(blobs[i][2][2]), int(blobs[i][2][3]))
                        cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
                        # cv2.putText(draw_frame, f"{i}", (x, y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    points2D_D = np.array(np.array(candidates).reshape(len(candidates), -1), dtype=np.float64)
                    points2D_U = np.array(cv2.undistortPoints(points2D_D, camera_matrix[0][0], camera_matrix[0][1])).reshape(-1, 2)

                    for perm in permutations(range(BLOB_CNT), window_size):
                        # print(perm)
                        points3D_perm = []
                        for idx in perm:
                            points3D_perm.append(MODEL_DATA[idx])

                        points3D = np.array(points3D_perm, dtype=np.float64)
                        METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P

                        # INPUT_ARRAY = [
                        #     CAM_ID,
                        #     points3D,
                        #     points2D_D if undistort == 0 else points2D_U,
                        #     camera_matrix[CAM_ID][0] if undistort == 0 else default_cameraK,
                        #     camera_matrix[CAM_ID][1] if undistort == 0 else default_dist_coeffs
                        # ]
                        # _, rvec, tvec, _ = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)                    


                    if AUTO_LOOP and UVC_MODE:
                        frame_cnt += 1
                                
            
        
            if AUTO_LOOP and UVC_MODE == 0:
                frame_cnt += 1
            # Display the resulting frame
            cv2.imshow('Frame', draw_frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('e'): 
                # Use 'e' key to exit the loop
                break
            elif key & 0xFF == ord('n'):
                if AUTO_LOOP and UVC_MODE == 0:
                    frame_cnt += 1     
            elif key & 0xFF == ord('b'):
                if AUTO_LOOP and UVC_MODE == 0:
                    frame_cnt -= 1    

        data = OrderedDict()
        data['BLOB_INFO'] = BLOB_INFO
        pickle_data(WRITE, 'BLOB_INFO.pickle', data)
        data = OrderedDict()
        data['CAMERA_INFO'] = CAMERA_INFO
        pickle_data(WRITE, 'CAMERA_INFO.pickle', data)

        # Release everything when done
        if UVC_MODE == 1:
            video.release()
        cv2.destroyAllWindows()
elif SOLUTION == 3:
    def check_pnp(points3D, points2D):
        STATUS = SUCCESS
        METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC
        INPUT_ARRAY = [
            CAM_ID,
            points3D,
            points2D,
            default_cameraK,
            default_dist_coeffs
        ]
        _, rvec, tvec, _ = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)

        # print('PnP_Solver rvec:', rvec, ' tvec:',  tvec)
        # rvec이나 tvec가 None인 경우 continue
        if rvec is None or tvec is None or rvec is NOT_SET or tvec is NOT_SET:
            STATUS = ERROR
        
        # rvec이나 tvec에 nan이 포함된 경우 continue
        if np.isnan(rvec).any() or np.isnan(tvec).any():
            STATUS = ERROR

        # rvec과 tvec의 모든 요소가 0인 경우 continue
        if np.all(rvec == 0) and np.all(tvec == 0):
            STATUS = ERROR
        # print('rvec:', rvec)
        # print('tvec:', tvec)                        

        if STATUS == SUCCESS:
            # RER = reprojection_error(points3D,
            #                         points2D,
            #                         rvec, tvec,
            #                         default_cameraK,
            #                         default_dist_coeffs)
            # # print('RER', RER)
            return 0, rvec, tvec
        else:
            return NOT_SET, NOT_SET, NOT_SET
    def auto_labeling():
        frame_cnt = 0
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
            image_files = sorted(glob.glob(os.path.join(script_dir,f"./render_img/rifts_right_9/test_1/" + '*.png'))) 
            # image_files = sorted(glob.glob(os.path.join(script_dir, f"./tmp/render/ARCTURAS/plane/" + '*.png'))) 
            # image_files = sorted(glob.glob(os.path.join(script_dir, f"./render_img/arcturas/test_1/" + '*.png')))        
            print('lenght of images: ', len(image_files))

        # Initialize each blob ID with a copy of the structure
        for blob_id in range(BLOB_CNT):
            BLOB_INFO[blob_id] = copy.deepcopy(BLOB_INFO_STRUCTURE)

        detect_time = []
        while video.isOpened() if UVC_MODE else True:
            print('\n')        
            if UVC_MODE:
                ret, frame_0 = video.read()
                filename = f"VIDEO Mode {camera_port}"
                if not ret:
                    break
            else:
                print(f"########## Frame {frame_cnt} ##########")
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

            blob_area = detect_led_lights(frame_0, TRACKER_PADDING)
            blobs = []

            for blob_id, bbox in enumerate(blob_area):
                (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                gcx, gcy, gsize = find_center(frame_0, (x, y, w, h))
                if gsize < BLOB_SIZE:
                    continue 
                
                cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
                blobs.append((gcx, gcy))

            blobs_sorted = np.array(sorted(blobs, key=lambda x:x[0]))
            # blobs_minus = np.array(sorted(blobs, key=lambda x:-x[0]))
            # points2D_D = []
            # for sorted_data in blobs_sorted:
            #     points2D_D.append([sorted_data[0], sorted_data[1]])            
            # points2D_D = np.array(points2D_D, dtype=np.float64)

            points2D_D = np.array(np.array(blobs_sorted).reshape(len(blobs_sorted), -1), dtype=np.float64)
            points2D_U = np.array(cv2.undistortPoints(points2D_D, camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][1])).reshape(-1, 2)
            # points2D_U = np.array(np.array(points2D_U).reshape(len(points2D_U), -1), dtype=np.float64)
            # points2D_D_R = np.flip(points2D_D, axis=0)
            # Alogirithm
            
            PNP_SOLVER_LENGTH = 4
            BLOBS_LENGTH = len(blobs_sorted)
            SEARCHING_WINDOW_SIZE = BLOBS_LENGTH * 2
            # print('BLOBS_LENGTH ', BLOBS_LENGTH)
            # print('points2D_D ', points2D_D)
            if BLOBS_LENGTH >= PNP_SOLVER_LENGTH:
                MIN_SOCRE = float('inf')
                MIN_INFO = {}
                MIN_POSE = []
                MIN_GROUP_ID = []
                MIN_POINTS3D = []
                
                start_time = time.time() 
                
                # SOLUTION 1
                # seen_combinations = set()
                # for grps in circular_sliding_window(range(BLOB_CNT), SEARCHING_WINDOW_SIZE): 
                #     for points3D_grp_comb in combinations(grps, BLOBS_LENGTH):
                #         # if points3D_grp_comb not in seen_combinations:
                #         #     seen_combinations.add(points3D_grp_comb)
                #             points3D_grp = MODEL_DATA[list(points3D_grp_comb), :]
                #             for points2d_u in sliding_window(points2D_U, BLOBS_LENGTH):
                #                 RER, RVEC, TVEC = check_pnp(points3D=points3D_grp, points2D=points2d_u)
                #                 if RER < MIN_SOCRE:
                #                     MIN_GROUP_ID = points3D_grp_comb
                #                     MIN_POINTS3D = points3D_grp
                #                     MIN_POSE = [RVEC, TVEC]
                
                #                 # Draw Blender projection
                # image_points, _ = cv2.projectPoints(MIN_POINTS3D,
                #                                     np.array(MIN_POSE[0]),
                #                                     np.array(MIN_POSE[1]),
                #                                     camera_matrix[CAM_ID][0],
                #                                     camera_matrix[CAM_ID][1])
                # # print('image_points', image_points)
                # image_points = image_points.reshape(-1, 2)
                # for idx, point in enumerate(image_points):
                #     # 튜플 형태로 좌표 변환
                #     pt = (int(point[0]), int(point[1]))
                #     cv2.circle(draw_frame, pt, 1, (255, 0, 0), -1)
                #     cv2.putText(draw_frame, str(MIN_GROUP_ID[idx]), (pt[0],pt[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                                    
                
                # SOLUTION 2
                # Advanced_Cython_Functions.cython_func(MODEL_DATA, points2D_U, BLOB_CNT, SEARCHING_WINDOW_SIZE, BLOBS_LENGTH)
                
                # SOLUTION 3
                # seen_combinations = set()
                # camera = {'model': 'SIMPLE_PINHOLE', 'width': 1280, 'height': 960, 'params': [715.159, 650.741, 489.184]}
                # for grps in circular_sliding_window(range(BLOB_CNT), SEARCHING_WINDOW_SIZE): 
                #     for points3D_grp_comb in combinations(grps, BLOBS_LENGTH):                        
                #         # if points3D_grp_comb not in seen_combinations:
                #         #     seen_combinations.add(points3D_grp_comb)
                #             points3D_grp = MODEL_DATA[list(points3D_grp_comb), :]
                #             for points2d_d in sliding_window(points2D_D, BLOBS_LENGTH):                                                        
                #                 pose, info = poselib.estimate_absolute_pose(points2d_d, points3D_grp, camera, {'max_reproj_error': 10.0}, {})
                #                 # print('pose ', pose)
                #                 # print('info ', info)
                #                 if info['model_score'] < MIN_SOCRE:
                #                     MIN_SOCRE = info['model_score']
                #                     MIN_POSE = pose
                #                     MIN_INFO = info
                #                     MIN_GROUP_ID = points3D_grp_comb
                #                     MIN_POINTS3D = points3D_grp
                
                # print('MIN SOCORE INFO')
                # print(MIN_INFO)
                # print(MIN_GROUP_ID)
                # if len(MIN_GROUP_ID) > 0:
                #     rvec, _ = cv2.Rodrigues(quat_to_rotm(MIN_POSE.q))
                #     image_points, _ = cv2.projectPoints(np.array(MIN_POINTS3D),
                #         np.array(rvec),
                #         np.array(MIN_POSE.t),
                #         camera_matrix[CAM_ID][0],
                #         camera_matrix[CAM_ID][1])
                #     image_points = image_points.reshape(-1, 2)
                #     for idx, point in enumerate(image_points):
                #         pt = (int(point[0]), int(point[1]))
                #         cv2.circle(draw_frame, pt, 1, (255, 255, 0), -1)
                #         cv2.putText(draw_frame, str(MIN_GROUP_ID[idx]), (pt[0],pt[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        
                
                # SOLUTION 4
                # seen_combinations = set()
                # for grps in circular_sliding_window(range(BLOB_CNT), SEARCHING_WINDOW_SIZE): 
                #     for points3D_grp_comb in combinations(grps, BLOBS_LENGTH):
                #         if points3D_grp_comb not in seen_combinations:
                #             seen_combinations.add(points3D_grp_comb)
                #             points3D_grp = MODEL_DATA[list(points3D_grp_comb), :]
                #             for points2d_u in sliding_window(points2D_U, BLOBS_LENGTH): 
                #                 X = np.array(points3D_grp)  
                #                 x = np.hstack((points2d_u, np.ones((points2d_u.shape[0], 1))))
                #                 # print('X ', X)
                #                 # print('x ', x)
                #                 poselib_result = poselib.p4pf(x, X, True)
                #                 # print('poselib_result ', poselib_result, ' len ', len(poselib_result[0]))
                #                 if len(poselib_result[0]) != 0:
                #                     rvec, _ = cv2.Rodrigues(quat_to_rotm(poselib_result[0][0].q))
                #                     # image_points, _ = cv2.projectPoints(np.array(points3D_grp),
                #                     #     np.array(rvec),
                #                     #     np.array(poselib_result[0][0].t),
                #                     #     camera_matrix[CAM_ID][0],
                #                     #     camera_matrix[CAM_ID][1])
                #                     # image_points = image_points.reshape(-1, 2)                                    
                #                     # for idx, point in enumerate(image_points):
                #                     #     pt = (int(point[0]), int(point[1]))
                #                     #     cv2.circle(draw_frame, pt, 1, (255, 255, 0), -1)
                #                     RER = reprojection_error(np.array(points3D_grp),
                #                                             points2d_u,
                #                                             rvec,
                #                                             poselib_result[0][0].t,
                #                                             default_cameraK,
                #                                             default_dist_coeffs)
                #                     if RER < MIN_SOCRE:
                #                         MIN_POSE = poselib_result[0][0]
                #                         MIN_GROUP_ID = points3D_grp_comb
                #                         MIN_POINTS3D = points3D_grp
                    
                # print('MIN SOCORE INFO')
                # print(MIN_INFO)
                # print(MIN_GROUP_ID)
                # if len(MIN_GROUP_ID) > 0:
                #     rvec, _ = cv2.Rodrigues(quat_to_rotm(MIN_POSE.q))
                #     image_points, _ = cv2.projectPoints(np.array(MIN_POINTS3D),
                #         np.array(rvec),
                #         np.array(MIN_POSE.t),
                #         camera_matrix[CAM_ID][0],
                #         camera_matrix[CAM_ID][1])
                #     image_points = image_points.reshape(-1, 2)
                #     for idx, point in enumerate(image_points):
                #         pt = (int(point[0]), int(point[1]))
                #         cv2.circle(draw_frame, pt, 1, (255, 255, 0), -1)
                #         cv2.putText(draw_frame, str(MIN_GROUP_ID[idx]), (pt[0],pt[1] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
                
                
                # SOLUTION 5                
                if BLOBS_LENGTH >= 6:
                    # Define default distortion coefficients and camera matrix
                    default_dist_coeffs = np.zeros((4, 1))
                    default_cameraK = np.eye(3).astype(np.float64)

                    # Use the default camera matrix as the intrinsic parameters
                    intrinsics_single = torch.tensor(np.array([default_cameraK]), dtype=torch.float64)

                    # intrinsics_single = torch.tensor([[[715.159, 0.0, 650.741],
                    #             [0.0, 715.159, 489.184],
                    #             [0.0, 0.0, 1.0]]], dtype=torch.float64)
                    # Convert to tensors
                    # MODEL_DATA_T = torch.tensor(MODEL_DATA, dtype=torch.float64)
                    # points2D_U_T = torch.tensor(points2D_U, dtype=torch.float64)
                    seen_combinations = set()

                    # Prepare lists for batch processing
                    batch_2d_points = []
                    batch_3d_points = []
                    batch_combinations = [] # <- New list for storing combinations
                    for grps in circular_sliding_window(range(BLOB_CNT), SEARCHING_WINDOW_SIZE):
                        for points3D_grp_comb in combinations(grps, BLOBS_LENGTH):
                            # Make the combination hashable by converting it to a tuple
                            points3D_grp_comb = tuple(sorted(points3D_grp_comb))
                            if points3D_grp_comb not in seen_combinations:
                                seen_combinations.add(points3D_grp_comb)
                                points3D_grp = MODEL_DATA[list(points3D_grp_comb), :]
                                for points2d_u in combinations(points2D_U, BLOBS_LENGTH):
                                    # Add to batch lists
                                    batch_2d_points.append(points2d_u)
                                    batch_3d_points.append(points3D_grp)
                                    batch_combinations.append(points3D_grp_comb) # <- Store the combination


                    # Convert lists of numpy arrays to single numpy arrays
                    batch_2d_points_np = np.array(batch_2d_points)
                    batch_3d_points_np = np.array(batch_3d_points)

                    # Convert numpy arrays to PyTorch tensors
                    batch_2d_points = torch.from_numpy(batch_2d_points_np)
                    batch_3d_points = torch.from_numpy(batch_3d_points_np)

                    # Duplicate intrinsics for each item in the batch
                    intrinsics = intrinsics_single.repeat(batch_2d_points.shape[0], 1, 1)

                    # Use tensors in kornia function
                    pred_world_to_cam = K.geometry.solve_pnp_dlt(batch_3d_points, batch_2d_points, intrinsics)
                    # Use the estimated world_to_cam matrices to project the 3d points back onto the image plane
                    
                    # Initialize the minimum reprojection error to a large value
                    min_reprojection_error = np.inf

                    # Initialize the best pose and 3D points to None
                    best_pose = None
                    best_3d_points = None
                    best_combination = None # <- New variable for the best combination
                    # For each predicted world_to_cam matrix...
                    for i in range(pred_world_to_cam.shape[0]):
                        # Unpack the rotation and translation vectors from the world_to_cam matrix
                        # print(pred_world_to_cam[i, :3, :3])
                        rvec = cv2.Rodrigues(pred_world_to_cam[i, :3, :3].cpu().numpy())[0]
                        tvec = pred_world_to_cam[i, :3, 3].cpu().numpy()

                        # Project the 3D points to the image plane using cv2.projectPoints
                        projected_2d_points, _ = cv2.projectPoints(batch_3d_points[i].cpu().numpy(), rvec, tvec, default_cameraK, default_dist_coeffs)
                        # Compute the reprojection error
                        reprojection_error = np.sum((projected_2d_points.squeeze() - batch_2d_points[i].cpu().numpy())**2)

                        # If this reprojection error is smaller than the current minimum...
                        if reprojection_error < min_reprojection_error:
                            # Update the minimum reprojection error
                            min_reprojection_error = reprojection_error

                            # Update the best pose and 3D points
                            best_pose = pred_world_to_cam[i]
                            best_3d_points = batch_3d_points[i]
                            best_combination = batch_combinations[i] # <- Update the best combination
 


                    print(f"Minimum reprojection error: {min_reprojection_error}")
                    print(f"Best pose: {best_pose}")
                    print(f"Best 3D points: {best_3d_points}")       
                    # After finding best pose and 3D points
                    rvec = cv2.Rodrigues(best_pose[:3, :3].cpu().numpy())[0]
                    tvec = best_pose[:3, 3].cpu().numpy()

                    # Project the 3D points to the image plane using cv2.projectPoints
                    image_points, _ = cv2.projectPoints(best_3d_points.cpu().numpy(), rvec, tvec, camera_matrix[0][0], camera_matrix[0][1])

                    # Reshape image_points for easier handling
                    image_points = image_points.reshape(-1, 2)

                    # Draw each point on the image
                    for idx, point in enumerate(image_points):
                        pt = (int(point[0]), int(point[1]))
                        cv2.circle(draw_frame, pt, 1, (255, 255, 0), -1)
                        cv2.putText(draw_frame, str(best_combination[idx]), (pt[0], pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)



                end_time = time.time()  # Store the current time again
                elapsed_time = end_time - start_time  # Calculate the difference
                print(f"The function took {elapsed_time} seconds to complete.")                                
                
                # detect_time.append(elapsed_time)

                # if len(detect_time) >= 10:
                #     print(f"Mean detect time {np.mean(detect_time)}")
                #     break
                

            if AUTO_LOOP and UVC_MODE == 0:
                frame_cnt += 1
            # Display the resulting frame
            cv2.imshow('Frame', draw_frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('e'): 
                # Use 'e' key to exit the loop
                break
            elif key & 0xFF == ord('n'):
                if AUTO_LOOP == 0 and UVC_MODE == 0:
                    frame_cnt += 1     
            elif key & 0xFF == ord('b'):
                if AUTO_LOOP == 0 and UVC_MODE == 0:
                    frame_cnt -= 1    

        # Release everything when done
        if UVC_MODE == 1:
            video.release()
        cv2.destroyAllWindows()
elif SOLUTION == 4:
    def init_trackers(trackers, frame):
        for id, data in trackers.items():
            tracker = cv2.TrackerCSRT_create()
            (x, y, w, h) = data['bbox']        
            ok = tracker.init(frame, (x - TRACKER_PADDING, y - TRACKER_PADDING, w + 2 * TRACKER_PADDING, h + 2 * TRACKER_PADDING))
            data['tracker'] = tracker
    def auto_labeling():
            frame_cnt = 0
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
                print('lenght of images: ', len(image_files))


            # Initialize each blob ID with a copy of the structure
            for blob_id in range(BLOB_CNT):
                BLOB_INFO[blob_id] = copy.deepcopy(BLOB_INFO_STRUCTURE)

            TRACKING_START = NOT_SET
            CURR_TRACKER = {}
            PREV_TRACKER = {}
            while video.isOpened() if UVC_MODE else True:
                print('\n')        
                if UVC_MODE:
                    ret, frame_0 = video.read()
                    filename = f"VIDEO Mode {camera_port}"
                    if not ret:
                        break
                else:
                    print(f"########## Frame {frame_cnt} ##########")
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
                blob_area = detect_led_lights(frame_0, TRACKER_PADDING)
                blobs = []
                bboxes = []
                for blob_id, bbox in enumerate(blob_area):
                    (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                    gcx, gcy, gsize = find_center(frame_0, (x, y, w, h))
                    if gsize < BLOB_SIZE:
                        continue 

                    cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
                    cv2.putText(draw_frame, f"{blob_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    blobs.append((gcx, gcy, bbox))     
                    bboxes.append({'idx': blob_id, 'bbox': bbox}) 
                
                if TRACKING_START == DONE:
                    CURR_TRACKER_CPY = CURR_TRACKER.copy()
                    for Tracking_ANCHOR, Tracking_DATA in CURR_TRACKER_CPY.items():
                        if Tracking_DATA['tracker'] is not None:
                            ret, (tx, ty, tw, th) = Tracking_DATA['tracker'].update(frame_0)
                            cv2.rectangle(draw_frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 1, 1)
                            cv2.putText(draw_frame, f"{Tracking_ANCHOR}", (tx, ty + th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            tcx, tcy, tsize = find_center(frame_0, (tx, ty, tw, th))
                            if Tracking_ANCHOR in PREV_TRACKER:
                                def check_distance(blob_centers, tcx, tcy):
                                    for center in blob_centers:
                                        gcx, gcy, _ = center
                                        distance = math.sqrt((gcx - tcx)**2 + (gcy - tcy)**2)
                                        if distance < 1:
                                            return False
                                    return True
                                dx = PREV_TRACKER[Tracking_ANCHOR][0] - tcx
                                dy = PREV_TRACKER[Tracking_ANCHOR][1] - tcy
                                euclidean_distance = math.sqrt(dx ** 2 + dy ** 2)
                                # 트랙커가 갑자기 이동
                                # 사이즈가 작은 경우
                                # 실패한 경우
                                # 중심점위치에 Blob_center 데이터가 없는 경우
                                exist_status = check_distance(blobs, tcx, tcy)
                                if exist_status or euclidean_distance > 10 or tsize < BLOB_SIZE or not ret:
                                    print('Tracker Broken')
                                    print('euclidean_distance:', euclidean_distance, ' tsize:', tsize, ' ret:', ret, 'exist_status:', exist_status)
                                    print('CUR_txy:', tcx, tcy)
                                    print('PRV_txy:', PREV_TRACKER[Tracking_ANCHOR])
                                    if ret == SUCCESS:
                                        del CURR_TRACKER[Tracking_ANCHOR]
                                        del PREV_TRACKER[Tracking_ANCHOR]
                                        # 여기서 PREV에 만들어진 위치를 집어넣어야 바로 안튕김
                                        print(f"tracker[{Tracking_ANCHOR}] deleted")
                                        continue
                                    else:
                                        break

                        PREV_TRACKER[Tracking_ANCHOR] = (tcx, tcy, (tx, ty, tw, th))

                if len(CURR_TRACKER) <= 0:
                    TRACKING_START = NOT_SET 

                if AUTO_LOOP and UVC_MODE == 0:
                    frame_cnt += 1
                # Display the resulting frame
                cv2.imshow('Frame', draw_frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('e'): 
                    # Use 'e' key to exit the loop
                    break
                elif key & 0xFF == ord('n'):
                    if AUTO_LOOP == 0 and UVC_MODE == 0:
                        frame_cnt += 1     
                elif key & 0xFF == ord('b'):
                    if AUTO_LOOP == 0 and UVC_MODE == 0:
                        frame_cnt -= 1
                elif key & 0xFF == ord('t'):
                    if TRACKING_START == NOT_SET:
                        print('bboxes:', bboxes)
                        if bboxes is None:
                            return            
                        for i in range(len(bboxes)):
                            CURR_TRACKER[bboxes[i]['idx']] = {'bbox': bboxes[i]['bbox'], 'tracker': None}
                        init_trackers(CURR_TRACKER, frame_0)          
                        TRACKING_START = DONE

            # Release everything when done
            if UVC_MODE == 1:
                video.release()
            cv2.destroyAllWindows()
def insert_ba_rt(**kwargs):
    print('insert_ba_rt START')
    BA_RT = pickle_data(READ, kwargs.get('ba_name'), None)['BA_RT']
    CAMERA_INFO = copy.deepcopy(pickle_data(READ, kwargs.get('camera_info_name'), None)['CAMERA_INFO'])
    BLOB_INFO = copy.deepcopy(pickle_data(READ, kwargs.get('blob_info_name'), None)['BLOB_INFO'])
    for frame_cnt, cam_info in CAMERA_INFO.items():
        LED_NUMBER = cam_info['LED_NUMBER']
        # print(f"{frame_cnt} {LED_NUMBER}")
        ba_rvec = BA_RT[int(frame_cnt)][:3]
        ba_tvec = BA_RT[int(frame_cnt)][3:]
        ba_rvec_reshape = np.array(ba_rvec).reshape(-1, 1)
        ba_tvec_reshape = np.array(ba_tvec).reshape(-1, 1)
        points3D = []
        for blob_id in LED_NUMBER:
            points3D.append(MODEL_DATA[int(blob_id)])               
            BLOB_INFO[blob_id]['BA_RT']['rt']['rvec'].append(ba_rvec_reshape)
            BLOB_INFO[blob_id]['BA_RT']['rt']['tvec'].append(ba_tvec_reshape)
            BLOB_INFO[blob_id]['BA_RT']['status'].append(DONE)
        CAMERA_INFO[f"{frame_cnt}"]['points3D'] = np.array(points3D, dtype=np.float64)
        CAMERA_INFO[f"{frame_cnt}"]['BA_RT']['rt']['rvec'] = ba_rvec_reshape
        CAMERA_INFO[f"{frame_cnt}"]['BA_RT']['rt']['tvec'] = ba_tvec_reshape

    data = OrderedDict()
    data['BLOB_INFO'] = BLOB_INFO
    pickle_data(WRITE, kwargs.get('blob_info_name'), data)

    data = OrderedDict()
    data['CAMERA_INFO'] = CAMERA_INFO
    pickle_data(WRITE, kwargs.get('camera_info_name'), data)
def insert_remake_3d():
    RIGID_3D_TRANSFORM_PCA = pickle_data(READ, 'RIGID_3D_TRANSFORM.pickle', None)['PCA_ARRAY_LSM']
    RIGID_3D_TRANSFORM_IQR = pickle_data(READ, 'RIGID_3D_TRANSFORM.pickle', None)['IQR_ARRAY_LSM']
    CAMERA_INFO = copy.deepcopy(pickle_data(READ, 'CAMERA_INFO.pickle', None)['CAMERA_INFO'])
    for frame_cnt, cam_info in CAMERA_INFO.items():
        LED_NUMBER = cam_info['LED_NUMBER']
        points3D_PCA = []
        points3D_IQR = []
        for blob_id in LED_NUMBER:
            points3D_PCA.append(RIGID_3D_TRANSFORM_PCA[int(blob_id)])
            points3D_IQR.append(RIGID_3D_TRANSFORM_IQR[int(blob_id)])
                
        points3D_PCA = np.array(points3D_PCA, dtype=np.float64)
        points3D_IQR = np.array(points3D_IQR, dtype=np.float64)
        CAMERA_INFO[f"{frame_cnt}"]['points3D_PCA'] = points3D_PCA
        CAMERA_INFO[f"{frame_cnt}"]['points3D_IQR'] = points3D_IQR

    data = OrderedDict()
    data['CAMERA_INFO'] = CAMERA_INFO
    pickle_data(WRITE, 'CAMERA_INFO.pickle', data)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(os.getcwd())

    # MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/arcturas_right.json"))
    MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/rifts_left_2.json"))
    BLOB_CNT = len(MODEL_DATA)

    # # Set the seed for Python's random module.
    # random.seed(1)
    # # Set the seed for NumPy's random module.
    # np.random.seed(1)
    # noise_std_dev = 0.0005 # Noise standard deviation. Adjust this value to your needs.
    # # Generate noise with the same shape as the original data.
    # noise = np.random.normal(scale=noise_std_dev, size=np.array(MODEL_DATA).shape)
    # # Add noise to the original data.
    # MODEL_DATA += noise

    TARGET_DEVICE = 'ARCUTRAS'
    combination_cnt = [4,5]
    print('PTS')
    for i, leds in enumerate(MODEL_DATA):
        print(f"{np.array2string(leds, separator=', ')},")

    from Advanced_Plot_3D import regenerate_pts_by_dist

    MODEL_DATA, DIRECTION = regenerate_pts_by_dist(0, MODEL_DATA, DIRECTION)
    # MODEL_DATA = np.array(MODEL_DATA)
    show_calibrate_data(np.array(MODEL_DATA), np.array(DIRECTION))

    auto_labeling()

    # from Advanced_Calibration import BA_RT, remake_3d_for_blob_info, LSM, draw_result, Check_Calibration_data_combination, init_plot
    # ax1, ax2 = init_plot(MODEL_DATA)
    # BA_RT(info_name='CAMERA_INFO.pickle', save_to='BA_RT.pickle', target='OPENCV') 
    # insert_ba_rt(camera_info_name='CAMERA_INFO.pickle', blob_info_name='BLOB_INFO.pickle', ba_name='BA_RT.pickle')
    # remake_3d_for_blob_info(blob_cnt=BLOB_CNT, info_name='BLOB_INFO.pickle', undistort=undistort, opencv=DONE, blender=NOT_SET, ba_rt=DONE)
    # LSM(TARGET_DEVICE, MODEL_DATA)
    # insert_remake_3d()
    # draw_result(MODEL_DATA, ax1=ax1, ax2=ax2, opencv=DONE, blender=NOT_SET, ba_rt=DONE, ba_3d=NOT_SET)
    # Check_Calibration_data_combination(combination_cnt)
    # plt.show()