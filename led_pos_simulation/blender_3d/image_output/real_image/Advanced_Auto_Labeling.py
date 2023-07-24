from Advanced_Function import *

camera_matrix = [
    [np.array([[712.623, 0.0, 653.448],
               [0.0, 712.623, 475.572],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)],
]

BLOB_SIZE = 150
TRACKER_PADDING = 3
CV_MAX_THRESHOLD = 255
CV_MIN_THRESHOLD = 150
MAX_LEVEL = 3
CAM_ID = 0

def sliding_window(data, window_size):
    for i in range(len(data) - window_size + 1):
        yield data[i:i + window_size]

def auto_labeling(camera_devices):
    # Select the first camera device
    camera_port = camera_devices[0]['port']

    # Open the video capture
    cap = cv2.VideoCapture(camera_port)

    # Set the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    recording = False
    if cap.isOpened():
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Unable to capture video")
                break
            draw_frame = frame.copy()
            _, frame_0 = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                   cv2.THRESH_TOZERO)
            center_x, center_y = 1280 // 2, 960 // 2
            cv2.line(draw_frame, (0, center_y), (1280, center_y), (255, 255, 255), 1)
            cv2.line(draw_frame, (center_x, 0), (center_x, 960), (255, 255, 255), 1) 


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

            if len(blobs) >= 3:
                CNT = len(blobs)
                window_size = 3
                test_cnt = 0
                for blob_idx in sliding_window(range(CNT), window_size):
                    if test_cnt >= 1:
                        break
                    # print(blob_idx)
                    candidates = []
                    for i in blob_idx:
                        candidates.append((blobs[i][0], blobs[i][1]))
                        (x, y, w, h) = (int(blobs[i][2][0]), int(blobs[i][2][1]), int(blobs[i][2][2]), int(blobs[i][2][3]))
                        cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 0), 1, 1)
                        cv2.putText(draw_frame, f"{i}", (x, y- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # points2D = np.array(candidates, dtype=np.float64)
                    points2D_D = np.array(np.array(candidates).reshape(len(candidates), -1), dtype=np.float64)
                    points2D_U = np.array(cv2.undistortPoints(points2D_D, camera_matrix[0][0], camera_matrix[0][1])).reshape(-1, 2)
                    # points2D_U = np.array(points2D_U.reshape(len(points2D_D), -1))
                    print('points2D_D\n', points2D_D)
                    print('points2D_U\n', points2D_U)
                    for model_idx in sliding_window(range(BLOB_CNT), window_size):
                        model_candidates = []
                        print(model_idx)
                        for i in model_idx:
                            print(i)
                            model_candidates.append(MODEL_DATA[i])
                        points3D = np.array(model_candidates, dtype=np.float64)
                        print('points3D\n', points3D)
                        points3D_reverse = np.flip(points3D, axis=0)
                        print('points3D_reverse\n', points3D_reverse)
                        MODEL_BRFS = [points3D, points3D_reverse]
                        for brfs in MODEL_BRFS:
                            X = np.array(brfs)
                            x = np.hstack((points2D_U, np.ones((points2D_U.shape[0], 1))))
                            print('X ', X)
                            print('x ', x)
                            poselib_result = poselib.p3p(x, X)
                            # print(X)
                            print(poselib_result)
                            
                            for solution_idx, pose in enumerate(poselib_result):
                                colors = [(255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
                                if is_valid_pose(pose):
                                    quat = pose.q
                                    tvec = pose.t
                                    rotm = quat_to_rotm(quat)
                                    rvec, _ = cv2.Rodrigues(rotm)
                                    print("PoseLib rvec: ", rvec.flatten(), ' tvec:', tvec)                               
                                    image_points, _ = cv2.projectPoints(np.array(brfs),
                                        np.array(rvec),
                                        np.array(tvec),
                                        camera_matrix[CAM_ID][0],
                                        camera_matrix[CAM_ID][1])
                                    image_points = image_points.reshape(-1, 2)
                                        
                                    for idx, point in enumerate(image_points):
                                        # 튜플 형태로 좌표 변환
                                        pt = (int(point[0]), int(point[1]))
                                        # print(pt)
                                        if pt[0] < 0 or pt[1] < 0:
                                            continue
                                        if pt[0] > 1280 or pt[1] > 960:
                                            continue
                                        cv2.circle(draw_frame, pt, 2, (0, 255, 0), -1)

                                        text_offset = (5, -5)
                                        text_pos = (pt[0] + text_offset[0], pt[1] + text_offset[1])
                                        cv2.putText(draw_frame, str(idx), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[solution_idx], 1, cv2.LINE_AA)
                    test_cnt += 1
            key = cv2.waitKey(1)
            if key & 0xFF == ord('e'): 
                # Use 'e' key to exit the loop
                break
            

            # Display the resulting frame
            cv2.imshow('Frame', draw_frame)

    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(os.getcwd())

    cam_dev_list = terminal_cmd('v4l2-ctl', '--list-devices')
    camera_devices = init_model_json(cam_dev_list)
    print(camera_devices)
    MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/rifts_left_2.json"))
    BLOB_CNT = len(MODEL_DATA)
    print('PTS')
    for i, leds in enumerate(MODEL_DATA):
        print(f"{np.array2string(leds, separator=', ')},")
    auto_labeling(camera_devices)