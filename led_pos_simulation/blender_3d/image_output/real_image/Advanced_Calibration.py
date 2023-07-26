from Advanced_Function import *
from connection.socket.socket_def import *


CAM_ID = 0

undistort = 1
TRACKING_ANCHOR_RECOGNIZE_SIZE = 1
max_level = 3
SHOW_PLOT = 1
FULL_COMBINATION_SEARCH = 0
CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960

HOPPING_CNT = 3
BLOB_CNT = -1

ANGLE = 3
VIDEO_MODE = 0
video_img_path = 'output_rifts_right_9.mkv'
blob_status = [[0, 'new'] for _ in range(BLOB_CNT)]

def init_trackers(trackers, frame):
    for id, data in trackers.items():
        tracker = cv2.TrackerCSRT_create()
        (x, y, w, h) = data['bbox']        
        ok = tracker.init(frame, (x - TRACKER_PADDING, y - TRACKER_PADDING, w + 2 * TRACKER_PADDING, h + 2 * TRACKER_PADDING))
        data['tracker'] = tracker
def mapping_id_blob(blob_centers, Tracking_ANCHOR, TRACKER):
    tcx = TRACKER[Tracking_ANCHOR][0]
    tcy = TRACKER[Tracking_ANCHOR][1]
    Tracking_ANCHOR = int(Tracking_ANCHOR)
    # Calculate distances for all blobs
    blob_distances = [(idx,
                       blob_centers[idx][0],
                       blob_centers[idx][1],  
                       math.sqrt((blob_centers[idx][0] - tcx) ** 2 + (blob_centers[idx][1] - tcy) ** 2),
					   -1)
                      for idx in range(len(blob_centers))]

    # Get blobs to the left of the tracker and sort them
    led_candidates_left = sorted(
        (blob for blob in blob_distances if blob[1] <= tcx),
        key=lambda blob: (-blob[1], blob[3])
    )
    # For blobs that are very close in x, prioritize the one closer to the tracker in y
    for i in range(len(led_candidates_left) - 1):
        if abs(led_candidates_left[i][1] - led_candidates_left[i + 1][1]) < TRACKING_ANCHOR_RECOGNIZE_SIZE:
            if led_candidates_left[i][3] > led_candidates_left[i + 1][3]:
                led_candidates_left[i], led_candidates_left[i + 1] = led_candidates_left[i + 1], led_candidates_left[i]
    # Do the same for blobs to the right of the tracker
    led_candidates_right = sorted(
        (blob for blob in blob_distances if blob[1] >= tcx),
        key=lambda blob: (blob[1], blob[3])
    )
    for i in range(len(led_candidates_right) - 1):
        if abs(led_candidates_right[i][1] - led_candidates_right[i + 1][1]) < TRACKING_ANCHOR_RECOGNIZE_SIZE:
            if led_candidates_right[i][3] > led_candidates_right[i + 1][3]:
                led_candidates_right[i], led_candidates_right[i + 1] = led_candidates_right[i + 1], led_candidates_right[i]
    ANCHOR_POS = LEDS_POSITION[Tracking_ANCHOR]
    clockwise = 0
    counterclockwise = 1


    def BLOB_ID_SEARCH(status, position, direction):
        # print(f"{status} {position} {direction}")
        COUNT = 1 
        BLOB_ID = -1
        while True:
            if direction == clockwise:
                POSITION_SEARCH = position + COUNT
                POSITION_SEARCH %= len(LEDS_POSITION)
                temp_id = LEDS_POSITION[POSITION_SEARCH]
                if status == TOP:                    
                    if temp_id != TOP:
                        COUNT += 1
                    else:
                        BLOB_ID = POSITION_SEARCH
                        break
                elif status == BOTTOM:
                    if temp_id != BOTTOM:
                        COUNT += 1
                    else:
                        BLOB_ID = POSITION_SEARCH
                        break          
            elif direction == counterclockwise:
                POSITION_SEARCH = position - COUNT
                if POSITION_SEARCH < 0:
                    POSITION_SEARCH += len(LEDS_POSITION)
                
                temp_id = LEDS_POSITION[POSITION_SEARCH]
                if status == TOP:                    
                    if temp_id != TOP:
                        COUNT += 1
                    else:
                        BLOB_ID = POSITION_SEARCH
                        break
                elif status == BOTTOM:
                    if temp_id != BOTTOM:
                        COUNT += 1
                    else:
                        BLOB_ID = POSITION_SEARCH
                        break
        return BLOB_ID                
    # Insert Blob ID here
    # print('BEFORE')
    # print('left_data')
    LEFT_BLOB_INFO = [-1, -1]
    led_candidates_left = [list(t) for t in led_candidates_left]
    for left_data in led_candidates_left:
        # print(left_data)
        if left_data[3] < TRACKING_ANCHOR_RECOGNIZE_SIZE:
            left_data[4] = Tracking_ANCHOR
            continue
        # TOP Searching and clockwise
        if ANCHOR_POS == TOP:
            if left_data[2] <= TOP_BOTTOM_LINE_Y:
                CURR_ID = Tracking_ANCHOR if LEFT_BLOB_INFO[TOP] == -1 else LEFT_BLOB_INFO[TOP]
                NEW_BLOB_ID = BLOB_ID_SEARCH(TOP, CURR_ID, clockwise if LEFT_RIGHT_DIRECTION == PLUS else counterclockwise)
                left_data[4] = NEW_BLOB_ID
                LEFT_BLOB_INFO[TOP] = NEW_BLOB_ID
        else:
            # BOTTOM Searching and clockwise
            if left_data[2] > TOP_BOTTOM_LINE_Y:
                CURR_ID = Tracking_ANCHOR if LEFT_BLOB_INFO[BOTTOM] == -1 else LEFT_BLOB_INFO[BOTTOM]
                NEW_BLOB_ID = BLOB_ID_SEARCH(BOTTOM, CURR_ID, clockwise if LEFT_RIGHT_DIRECTION == PLUS else counterclockwise)
                left_data[4] = NEW_BLOB_ID
                LEFT_BLOB_INFO[BOTTOM] = NEW_BLOB_ID
        
    # print('right_data')
    led_candidates_right = [list(t) for t in led_candidates_right]
    RIGHT_BLOB_INFO = [-1, -1]
    for right_data in led_candidates_right:
        # print(right_data)
        if right_data[3] < TRACKING_ANCHOR_RECOGNIZE_SIZE:
            right_data[4] = Tracking_ANCHOR
            continue

        if ANCHOR_POS == TOP:
            if right_data[2] <= TOP_BOTTOM_LINE_Y:
                CURR_ID = Tracking_ANCHOR if RIGHT_BLOB_INFO[TOP] == -1 else RIGHT_BLOB_INFO[TOP]
                NEW_BLOB_ID = BLOB_ID_SEARCH(TOP, CURR_ID, counterclockwise if LEFT_RIGHT_DIRECTION == PLUS else clockwise)
                right_data[4] = NEW_BLOB_ID
                RIGHT_BLOB_INFO[TOP] = copy.deepcopy(NEW_BLOB_ID)
        else:
            if right_data[2] > TOP_BOTTOM_LINE_Y:
                CURR_ID = Tracking_ANCHOR if RIGHT_BLOB_INFO[BOTTOM] == -1 else RIGHT_BLOB_INFO[BOTTOM]
                NEW_BLOB_ID = BLOB_ID_SEARCH(BOTTOM, CURR_ID, counterclockwise if LEFT_RIGHT_DIRECTION == PLUS else clockwise)
                right_data[4] = NEW_BLOB_ID
                RIGHT_BLOB_INFO[BOTTOM] = copy.deepcopy(NEW_BLOB_ID)

    # Remove rows where the 4th value is -1 or the 3rd value is less than or equal to 1
    led_candidates_left = [row for row in led_candidates_left if row[4] != -1]
    led_candidates_right = [row for row in led_candidates_right if row[4] != -1]

    # print('AFTER')
    # print('left_data')
    # for left_data in led_candidates_left:
    #     print(left_data)
    # print('right_data')
    # for right_data in led_candidates_right:
    #     print(right_data)

    return led_candidates_left, led_candidates_right
def blob_setting(script_dir, SERVER, blob_file):
    print('blob_setting START')

    bboxes = []
    json_file = os.path.join(script_dir, blob_file)
    json_data = rw_json_data(READ, json_file, None)
    if json_data != ERROR:
        bboxes = json_data['bboxes']
    if SERVER == 1:
        return bboxes
    
    image_files = sorted(glob.glob(os.path.join(script_dir, camera_img_path + '*.png')))
    # camera_params = read_camera_log(os.path.join(script_dir, camera_log_path))
    if VIDEO_MODE == 1:
        video = cv2.VideoCapture(video_img_path)
    frame_cnt = 0
    while video.isOpened() if VIDEO_MODE else True:
        if VIDEO_MODE == 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_cnt)  # Frame indices start from 0
            ret, frame = video.read()
            filename = f"VIDEO Mode {video_img_path}"
            video.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)
            if not ret:
                break
        else:
            if frame_cnt >= len(image_files):
                break
            frame = cv2.imread(image_files[frame_cnt])
            filename = f"IMAGE Mode {os.path.basename(image_files[frame_cnt])}"
            if frame is None:
                print("Cannot read the first image")
                cv2.destroyAllWindows()
                exit()
            frame = cv2.resize(frame, (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT))
        

        _, frame = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                   cv2.THRESH_TOZERO)
    
        draw_frame = frame.copy()
        height, width = draw_frame.shape
        center_x, center_y = width // 2, height // 2
        cv2.line(draw_frame, (0, center_y), (width, center_y), (255, 0, 0), 1)
        cv2.line(draw_frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)

        # brvec, btvec = camera_params[frame_cnt + 1]
        
        cv2.putText(draw_frame, f"frame_cnt {frame_cnt} [{filename}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        # cv2.putText(draw_frame, f"rvec: {brvec}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.putText(draw_frame, f"tvec: {btvec}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        blob_area = detect_led_lights(frame, TRACKER_PADDING, 5, 500)
        cv2.namedWindow('image')
        partial_click_event = functools.partial(click_event, frame=frame, blob_area_0=blob_area, bboxes=bboxes)
        cv2.setMouseCallback('image', partial_click_event)
        key = cv2.waitKey(1)

        if key == ord('c'):
            print('clear area')
            bboxes.clear()
        elif key == ord('s'):
            print('save blob area')
            json_data = OrderedDict()
            json_data['bboxes'] = bboxes
            # Write json data
            rw_json_data(WRITE, json_file, json_data)

        elif key & 0xFF == 27:
            print('ESC pressed')
            cv2.destroyAllWindows()
            sys.exit()
            return

        elif key == ord('q'):
            print('go next step')
            break
        elif key == ord('n'):
            frame_cnt += 1
            bboxes.clear()
            print('go next IMAGE')
        elif key == ord('b'):
            frame_cnt -= 1
            bboxes.clear()
            print('go back IMAGE')

        draw_blobs_and_ids(draw_frame, blob_area, bboxes)
        if SERVER == 0:
            cv2.imshow('image', draw_frame)

    print('done')
    cv2.destroyAllWindows()

    return bboxes
def init_new_tracker(prev_frame, Tracking_ANCHOR, CURR_TRACKER, PREV_TRACKER):
    # prev_frame = cv2.imread(frame)
    if prev_frame is None or prev_frame.size == 0:
        print(f"Failed to load prev frame")
        return ERROR, None
    # draw_frame = prev_frame.copy()
    _, prev_frame = cv2.threshold(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                  cv2.THRESH_TOZERO)
    # filename = os.path.basename(frame)
    # cv2.putText(draw_frame, f"{filename}", (draw_frame.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 255, 255), 1)
    # find Blob area by findContours
    blob_area = detect_led_lights(prev_frame, TRACKER_PADDING, 5, 500)
    blob_centers = []
    for blob_id, bbox in enumerate(blob_area):
        (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        gcx, gcy, gsize = find_center(prev_frame, (x, y, w, h))
        if gsize < BLOB_SIZE:
            continue
        # cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
        # cv2.putText(draw_frame, f"{blob_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        blob_centers.append((gcx, gcy, bbox))
    prev_pos_candidates, _ = mapping_id_blob(blob_centers, Tracking_ANCHOR, PREV_TRACKER)
    print('prev_pos_candidates:', prev_pos_candidates[0])
    # for bid, data in enumerate(blob_centers):
    #     print(bid, ':', data)

    NEW_Tracking_ANCHOR = -1
    NEW_Tracking_bbox = None    

    for i in range(min(HOPPING_CNT, len(prev_pos_candidates))):
        NEW_BLOB_ID = prev_pos_candidates[i][4]
        NEW_Tracking_bbox = blob_centers[prev_pos_candidates[i][0]][2]
        NEW_Tracking_ANCHOR = NEW_BLOB_ID
    if NEW_Tracking_ANCHOR != -1 and NEW_Tracking_bbox is not None:
        CURR_TRACKER_CPY = CURR_TRACKER.copy()
        for Other_side_Tracking_ANCHOR, _ in CURR_TRACKER_CPY.items():
            if Other_side_Tracking_ANCHOR != Tracking_ANCHOR:
                Other_Tracking_bbox = PREV_TRACKER[Other_side_Tracking_ANCHOR][2]
                CURR_TRACKER[Other_side_Tracking_ANCHOR] = {'bbox': Other_Tracking_bbox, 'tracker': None}
                init_trackers(CURR_TRACKER, prev_frame)

        # print(f"UPDATE NEW_Tracking_ANCHOR {NEW_Tracking_ANCHOR} NEW_Tracking_bbox {NEW_Tracking_bbox}")
        CURR_TRACKER[NEW_Tracking_ANCHOR] = {'bbox': NEW_Tracking_bbox, 'tracker': None}
        init_trackers(CURR_TRACKER, prev_frame)
        # tcx, tcy, _ = find_center(prev_frame, NEW_Tracking_bbox)
        # PREV_TRACKER[NEW_Tracking_ANCHOR] = (tcx, tcy)
        # cv2.imshow('TRACKER change', draw_frame)
        # cv2.waitKey(0)
        
        return SUCCESS, CURR_TRACKER
    else:
        return ERROR, None
def gathering_data_single(ax1, script_dir, bboxes, start, end, DO_CALIBRATION_TEST = 0, DO_BA = 0):
    print('gathering_data_single START')
    BA_RT = pickle_data(READ, 'BA_RT.pickle', None)['BA_RT']
    RIGID_3D_TRANSFORM_PCA = pickle_data(READ, 'RIGID_3D_TRANSFORM.pickle', None)['PCA_ARRAY_LSM']
    RIGID_3D_TRANSFORM_IQR = pickle_data(READ, 'RIGID_3D_TRANSFORM.pickle', None)['IQR_ARRAY_LSM']

    camera_params = read_camera_log(os.path.join(script_dir, camera_log_path))
    image_files = sorted(glob.glob(os.path.join(script_dir, camera_img_path + '*.png')))
    frame_cnt = 0

    CURR_TRACKER = {}
    PREV_TRACKER = {}

    # Init Multi Tracker
    TRACKING_START = NOT_SET
    # trackerType = "CSRT"
    # multiTracker = cv2.legacy.MultiTracker_create()

    mutex = threading.Lock()
    # Initialize each blob ID with a copy of the structure
    for blob_id in range(BLOB_CNT):
        BLOB_INFO[blob_id] = copy.deepcopy(BLOB_INFO_STRUCTURE)

    if VIDEO_MODE == 1:
        video = cv2.VideoCapture(video_img_path)
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        print('lenght of images: ', total_frames)
    else:
        print('lenght of images: ', len(image_files))


    while video.isOpened() if VIDEO_MODE else True:
        print('\n')
        print(f"########## FraSme {frame_cnt} ##########")
        if VIDEO_MODE == 1:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_cnt)
            ret, frame_0 = video.read()
            filename = f"VIDEO Mode {video_img_path}"
            video.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)
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

        if frame_cnt < start:
            print(f"skip frame_cnt{frame_cnt}")
            frame_cnt += 1
            if SERVER == 0:
                cv2.imshow("Tracking", draw_frame)
            key = cv2.waitKey(1)
            continue

        if frame_cnt > end + 1:
            break

        if TRACKING_START == NOT_SET:
            print('bboxes:', bboxes)
            if bboxes is None:
                return            
            for i in range(len(bboxes)):
                CURR_TRACKER[bboxes[i]['idx']] = {'bbox': bboxes[i]['bbox'], 'tracker': None}
            init_trackers(CURR_TRACKER, frame_0)
            # for i, data in enumerate(bboxes):
            #     multiTracker.add(createTrackerByName(trackerType), frame_0, data['bbox'])
        TRACKING_START = DONE

        height, width = frame_0.shape
        center_x, center_y = width // 2, height // 2
        cv2.line(draw_frame, (0, center_y), (width, center_y), (255, 255, 255), 1)
        cv2.line(draw_frame, (center_x, 0), (center_x, height), (255, 255, 255), 1)   
        cv2.line(draw_frame, (0, TOP_BOTTOM_LINE_Y), (width, TOP_BOTTOM_LINE_Y), (0, 255, 0), 1)     

        # find Blob area by findContours
        blob_area = detect_led_lights(frame_0, TRACKER_PADDING, 5, 500)
        blob_centers = []    
        for blob_id, bbox in enumerate(blob_area):
            (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            gcx, gcy, gsize = find_center(frame_0, (x, y, w, h))
            if gsize < BLOB_SIZE:
                continue
            
            if DO_PYRAMID == 1:
                overlapping = check_blobs_with_pyramid(frame_0, draw_frame, x, y, w, h, max_level)
                if overlapping == True:
                    cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 0, 255), 1, 1)
                    cv2.putText(draw_frame, f"SEG", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    continue

            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
            # cv2.putText(draw_frame, f"{blob_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # cv2.putText(draw_frame, f"{int(gcx)},{int(gcy)},{gsize}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
            #             (255, 255, 255), 1)
            blob_centers.append((gcx, gcy, bbox))
            # print(f"{blob_id} : {gcx}, {gcy}")

        CURR_TRACKER_CPY = CURR_TRACKER.copy()
        # print('CURR_TRACKER_CPY', CURR_TRACKER_CPY)

        if len(CURR_TRACKER_CPY) > 0:
            TEMP_BLOBS = {}
            TRACKER_BROKEN_STATUS = NOT_SET
            for Tracking_ANCHOR, Tracking_DATA in CURR_TRACKER_CPY.items():
                if Tracking_DATA['tracker'] is not None:
                    # print('Tracking_ANCHOR:', Tracking_ANCHOR)
                    ret, (tx, ty, tw, th) = Tracking_DATA['tracker'].update(frame_0)
                    # cv2.rectangle(draw_frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 1, 1)                    
                    cv2.putText(draw_frame, f'{Tracking_ANCHOR}', (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                    tcx, tcy, tsize = find_center(frame_0, (tx, ty, tw, th))
                    cv2.circle(draw_frame, (int(tcx), int(tcy)), 2, (0, 255, 0), -1)
                    if Tracking_ANCHOR in PREV_TRACKER:
                        def check_distance(blob_centers, tcx, tcy):
                            for center in blob_centers:
                                gcx, gcy, _ = center
                                distance = ((gcx - tcx)**2 + (gcy - tcy)**2)**0.5
                                if distance < TRACKING_ANCHOR_RECOGNIZE_SIZE:
                                    return False
                            return True
                        dx = PREV_TRACKER[Tracking_ANCHOR][0] - tcx
                        dy = PREV_TRACKER[Tracking_ANCHOR][1] - tcy
                        euclidean_distance = math.sqrt(dx ** 2 + dy ** 2)
                        # 트랙커가 갑자기 이동
                        # 사이즈가 작은 경우
                        # 실패한 경우
                        # 중심점위치에 Blob_center 데이터가 없는 경우
                        exist_status = check_distance(blob_centers, tcx, tcy)
                        if exist_status or euclidean_distance > THRESHOLD_DISTANCE or tsize < BLOB_SIZE or not ret:
                            print('Tracker Broken')
                            print('euclidean_distance:', euclidean_distance, ' tsize:', tsize, ' ret:', ret, 'exist_status:', exist_status)
                            print('CUR_txy:', tcx, tcy)
                            print('PRV_txy:', PREV_TRACKER[Tracking_ANCHOR])
                            # del CURR_TRACKER[Tracking_ANCHOR]
                            if VIDEO_MODE:
                                video.set(cv2.CAP_PROP_POS_FRAMES, frame_cnt - 1)
                                ret, search_frame = video.read()
                            else:
                                search_frame =  image_files[frame_cnt - 1]
                                search_frame = cv2.imread(search_frame)      
                
                            ret, CURR_TRACKER = init_new_tracker(search_frame, Tracking_ANCHOR, CURR_TRACKER, PREV_TRACKER)

                            if ret == SUCCESS:
                                del CURR_TRACKER[Tracking_ANCHOR]
                                del PREV_TRACKER[Tracking_ANCHOR]
                                # 여기서 PREV에 만들어진 위치를 집어넣어야 바로 안튕김
                                print(f"tracker[{Tracking_ANCHOR}] deleted")
                                frame_cnt -= 1
                                TRACKER_BROKEN_STATUS = DONE
                                break
                            else:
                                print('Tracker Change Failed')
                                if SERVER == 0:
                                    cv2.imshow("Tracking", draw_frame)
                                while True:
                                    key = cv2.waitKey(1)
                                    if key & 0xFF == ord('q'):
                                        break
                                break

                    PREV_TRACKER[Tracking_ANCHOR] = (tcx, tcy, (tx, ty, tw, th))
                    led_candidates_left, led_candidates_right = mapping_id_blob(blob_centers, Tracking_ANCHOR, PREV_TRACKER) 

                    for i in range(len(led_candidates_left)):
                        NEW_BLOB_ID = led_candidates_left[i][4]
                        (cx, cy, cw, ch) = blob_centers[led_candidates_left[i][0]][2]                        
                        cv2.rectangle(draw_frame, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 1, 1)
                        if NEW_BLOB_ID != int(Tracking_ANCHOR):
                            cv2.putText(draw_frame, f'{NEW_BLOB_ID}', (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 255, 255), 1)
                        points2D_D = np.array([blob_centers[led_candidates_left[i][0]][0], blob_centers[led_candidates_left[i][0]][1]], dtype=np.float64)
                        points2D_U = np.array(cv2.undistortPoints(points2D_D, camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][1])).reshape(-1, 2)
                        TEMP_BLOBS[NEW_BLOB_ID] = {'D': [blob_centers[led_candidates_left[i][0]][0], blob_centers[led_candidates_left[i][0]][1]],
                        'U': points2D_U} 

                    for i in range(len(led_candidates_right)):
                        NEW_BLOB_ID = led_candidates_right[i][4]
                        (cx, cy, cw, ch) = blob_centers[led_candidates_right[i][0]][2]
                        cv2.rectangle(draw_frame, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 1, 1)
                        if NEW_BLOB_ID != int(Tracking_ANCHOR):
                            cv2.putText(draw_frame, f'{NEW_BLOB_ID}', (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1) 
                        points2D_D = np.array([blob_centers[led_candidates_right[i][0]][0], blob_centers[led_candidates_right[i][0]][1]], dtype=np.float64)
                        points2D_U = np.array(cv2.undistortPoints(points2D_D, camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][1])).reshape(-1, 2)
                        TEMP_BLOBS[NEW_BLOB_ID] = {'D': [blob_centers[led_candidates_right[i][0]][0], blob_centers[led_candidates_right[i][0]][1]],
                        'U': points2D_U}
                else:
                    print(f"No tracker initialized for id: {Tracking_ANCHOR}")
                    break
            

            if TRACKER_BROKEN_STATUS == DONE:
                print(f"{frame_cnt} rollback")
                continue


            '''            
            Algorithm Added            
            '''

            blob_status = SUCCESS
            camera_params_pos = frame_cnt + 1
            # camera_params_pos = frame_cnt - start + 1
            # camera_params_pos = (camera_params_pos - 1) * ANGLE + 1  # Update this line
            if camera_params_pos <= len(camera_params):                
                brvec, btvec = camera_params[camera_params_pos]
                brvec_reshape = np.array(brvec).reshape(-1, 1)
                btvec_reshape = np.array(btvec).reshape(-1, 1)
                # print('Blender rvec:', brvec_reshape.flatten(), ' tvec:', btvec_reshape.flatten())

                CAMERA_INFO[f"{frame_cnt}"] = copy.deepcopy(CAMERA_INFO_STRUCTURE)
                LED_NUMBER = []
                points2D = []
                points2D_U = []
                points3D = []
                
                # TEST CODE
                points3D_PCA = []
                points3D_IQR = []            
                
                TEMP_BLOBS = OrderedDict(sorted(TEMP_BLOBS.items(), key=lambda t: t[0], reverse=True))
                for blob_id, blob_data in TEMP_BLOBS.items():
                    LED_NUMBER.append(int(blob_id))
                    points2D.append(blob_data['D'])
                    points2D_U.append(blob_data['U'])
                    
                    BLOB_INFO[blob_id]['points2D_D']['greysum'].append(blob_data['D'])
                    BLOB_INFO[blob_id]['points2D_U']['greysum'].append(blob_data['U'])
                    BLOB_INFO[blob_id]['BLENDER']['rt']['rvec'].append(brvec_reshape)
                    BLOB_INFO[blob_id]['BLENDER']['rt']['tvec'].append(btvec_reshape)
                    BLOB_INFO[blob_id]['BLENDER']['status'].append(DONE)
                    
                    points3D.append(MODEL_DATA[int(blob_id)])
                    if DO_CALIBRATION_TEST == 1:
                        # points3D_PCA.append(calibrated_led_data_PCA[int(blob_id)])
                        # points3D_IQR.append(calibrated_led_data_IQR[int(blob_id)])
                        points3D_PCA.append(RIGID_3D_TRANSFORM_PCA[int(blob_id)])
                        points3D_IQR.append(RIGID_3D_TRANSFORM_IQR[int(blob_id)])
                
                # print('START Pose Estimation')
                points2D = np.array(np.array(points2D).reshape(len(points2D), -1), dtype=np.float64)
                points2D_U = np.array(np.array(points2D_U).reshape(len(points2D_U), -1), dtype=np.float64)
                points3D = np.array(points3D, dtype=np.float64)
                if DO_CALIBRATION_TEST == 1:
                    points3D_PCA = np.array(points3D_PCA, dtype=np.float64)
                    points3D_IQR = np.array(points3D_IQR, dtype=np.float64)
                # print('LED_NUMBER: ', LED_NUMBER)
                # print('points2D\n', points2D)
                # print('points2D_U\n', points2D_U)
                # print('points3D\n', points3D)

                                    
                # Make CAMERA_INFO data for check rt STD
                CAMERA_INFO[f"{frame_cnt}"]['points3D'] = points3D
                if DO_CALIBRATION_TEST == 1:
                    CAMERA_INFO[f"{frame_cnt}"]['points3D_PCA'] = points3D_PCA
                    CAMERA_INFO[f"{frame_cnt}"]['points3D_IQR'] = points3D_IQR
                    
                CAMERA_INFO[f"{frame_cnt}"]['points2D']['greysum'] = points2D
                CAMERA_INFO[f"{frame_cnt}"]['points2D_U']['greysum'] = points2D_U            
                CAMERA_INFO[f"{frame_cnt}"]['LED_NUMBER'] =LED_NUMBER
                CAMERA_INFO[f"{frame_cnt}"]['BLENDER']['rt']['rvec'] = brvec_reshape
                CAMERA_INFO[f"{frame_cnt}"]['BLENDER']['rt']['tvec'] = btvec_reshape

                if DO_BA == 1:
                    ########################### BUNDLE ADJUSTMENT RT #######################
                    ba_rvec = BA_RT[frame_cnt - start][:3]
                    ba_tvec = BA_RT[frame_cnt - start][3:]
                    ba_rvec_reshape = np.array(ba_rvec).reshape(-1, 1)
                    ba_tvec_reshape = np.array(ba_tvec).reshape(-1, 1)
                    print('ba_rvec : ', ba_rvec)
                    print('ba_tvec : ', ba_tvec)
                    image_points, _ = cv2.projectPoints(points3D,
                                                        np.array(ba_rvec),
                                                        np.array(ba_tvec),
                                                        camera_matrix[CAM_ID][0],
                                                        camera_matrix[CAM_ID][1])
                    image_points = image_points.reshape(-1, 2)

                    for point in image_points:
                        pt = (int(point[0]), int(point[1]))
                        cv2.circle(draw_frame, pt, 2, (0, 0, 0), -1)
    
                    for blob_id in LED_NUMBER:
                        BLOB_INFO[blob_id]['BA_RT']['rt']['rvec'].append(ba_rvec_reshape)
                        BLOB_INFO[blob_id]['BA_RT']['rt']['tvec'].append(ba_tvec_reshape)
                        BLOB_INFO[blob_id]['BA_RT']['status'].append(DONE)
                    CAMERA_INFO[f"{frame_cnt}"]['BA_RT']['rt']['rvec'] = ba_rvec_reshape
                    CAMERA_INFO[f"{frame_cnt}"]['BA_RT']['rt']['tvec'] = ba_tvec_reshape
                    ########################### BUNDLE ADJUSTMENT RT #######################

                LENGTH = len(LED_NUMBER)
                if LENGTH >= 4:
                    # print('PnP Solver OpenCV')
                    if LENGTH >= 5:
                        METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC
                    elif LENGTH == 4:
                        METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P
                    INPUT_ARRAY = [
                        CAM_ID,
                        points3D,
                        points2D if undistort == 0 else points2D_U,
                        camera_matrix[CAM_ID][0] if undistort == 0 else default_cameraK,
                        camera_matrix[CAM_ID][1] if undistort == 0 else default_dist_coeffs
                    ]
                    ret, rvec, tvec, _ = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
                    rvec_reshape = np.array(rvec).reshape(-1, 1)
                    tvec_reshape = np.array(tvec).reshape(-1, 1)
                    print('PnP_Solver rvec:', rvec.flatten(), ' tvec:',  tvec.flatten())
                    for blob_id in LED_NUMBER:
                        BLOB_INFO[blob_id]['OPENCV']['rt']['rvec'].append(rvec_reshape)
                        BLOB_INFO[blob_id]['OPENCV']['rt']['tvec'].append(tvec_reshape)
                        BLOB_INFO[blob_id]['OPENCV']['status'].append(DONE)

                    # Draw OpenCV projection
                    image_points, _ = cv2.projectPoints(points3D,
                                                        rvec_reshape,
                                                        tvec_reshape,
                                                        camera_matrix[CAM_ID][0],
                                                        camera_matrix[CAM_ID][1])
                    image_points = image_points.reshape(-1, 2)

                    for point in image_points:
                        # 튜플 형태로 좌표 변환
                        pt = (int(point[0]), int(point[1]))
                        cv2.circle(draw_frame, pt, 1, (0, 0, 255), -1)
                    
                    # Draw Blender projection
                    image_points, _ = cv2.projectPoints(points3D,
                                                        np.array(brvec_reshape),
                                                        np.array(btvec_reshape),
                                                        camera_matrix[CAM_ID][0],
                                                        camera_matrix[CAM_ID][1])
                    image_points = image_points.reshape(-1, 2)

                    for point in image_points:
                        # 튜플 형태로 좌표 변환
                        pt = (int(point[0]), int(point[1]))
                        cv2.circle(draw_frame, pt, 1, (255, 0, 0), -1)                        

                    CAMERA_INFO[f"{frame_cnt}"]['OPENCV']['rt']['rvec'] = rvec_reshape
                    CAMERA_INFO[f"{frame_cnt}"]['OPENCV']['rt']['tvec'] = tvec_reshape

                elif LENGTH == 3:
                    if DO_P3P == 1:
                        #P3P
                        mutex.acquire()
                        try:
                            print('P3P LamdaTwist')
                            points2D_U = np.array(points2D_U.reshape(len(points2D), -1))                   
                            X = np.array(points3D)   
                            x = np.hstack((points2D_U, np.ones((points2D_U.shape[0], 1))))
                            print('X ', X)
                            print('x ', x)
                            poselib_result = poselib.p3p(x, X)
                            visible_detection = NOT_SET
                            for solution_idx, pose in enumerate(poselib_result):
                                colors = [(255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
                                colorstr = ['blue', 'green', 'purple', 'yellow']
                                if is_valid_pose(pose):
                                    quat = pose.q
                                    tvec = pose.t
                                    rotm = quat_to_rotm(quat)
                                    rvec, _ = cv2.Rodrigues(rotm)
                                    print("PoseLib rvec: ", rvec.flatten(), ' tvec:', tvec)                               
                                    image_points, _ = cv2.projectPoints(np.array(MODEL_DATA),
                                        np.array(rvec),
                                        np.array(tvec),
                                        camera_matrix[CAM_ID][0],
                                        camera_matrix[CAM_ID][1])
                                    image_points = image_points.reshape(-1, 2)
                                    # print('image_points\n', image_points)
                                    cam_pos, cam_dir, _ = calculate_camera_position_direction(rvec, tvec)
                                    ax1.scatter(*cam_pos, c=colorstr[solution_idx], marker='o', label=f"POS{solution_idx}")
                                    ax1.quiver(*cam_pos, *cam_dir, color=colorstr[solution_idx], label=f"DIR{solution_idx}", length=0.1)    
                                    
                                    ###############################            
                                    visible_result = check_angle_and_facing(MODEL_DATA, DIRECTION, cam_pos, quat, LED_NUMBER)
                                    # print('visible_result:', visible_result)
                                    visible_status = SUCCESS
                                    for blob_id, status in visible_result.items():
                                        if status == False:
                                            visible_status = ERROR
                                            print(f"{solution_idx} pose unvisible led {blob_id}")
                                            break                                
                                    if visible_status == SUCCESS:
                                        visible_detection = DONE
                                        for blob_id in LED_NUMBER:
                                            BLOB_INFO[blob_id]['OPENCV']['rt']['rvec'].append(np.array(rvec).reshape(-1, 1))
                                            BLOB_INFO[blob_id]['OPENCV']['rt']['tvec'].append(np.array(tvec).reshape(-1, 1))
                                            BLOB_INFO[blob_id]['OPENCV']['status'].append(DONE)
                                    ###############################
                                        
                                    for idx, point in enumerate(image_points):
                                        # 튜플 형태로 좌표 변환
                                        pt = (int(point[0]), int(point[1]))
                                        if idx in LED_NUMBER:
                                            cv2.circle(draw_frame, pt, 2, (0, 0, 255), -1)
                                        else:
                                            cv2.circle(draw_frame, pt, 1, colors[solution_idx], -1)
                                        
                                        text_offset = (5, -5)
                                        text_pos = (pt[0] + text_offset[0], pt[1] + text_offset[1])
                                        cv2.putText(draw_frame, str(idx), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[solution_idx], 1, cv2.LINE_AA)

                            if visible_detection == NOT_SET:
                                blob_status = ERROR
                        finally:
                            mutex.release()
                    else:
                        blob_status = ERROR
                else:
                    blob_status = ERROR
                    print('NOT Enough blobs')                                                             

                
                if blob_status == ERROR:
                    for blob_id in LED_NUMBER:
                        # Use an 'empty' numpy array as our NOT_SET value
                        BLOB_INFO[blob_id]['OPENCV']['rt']['rvec'].append(NOT_SET)
                        BLOB_INFO[blob_id]['OPENCV']['rt']['tvec'].append(NOT_SET)
                        BLOB_INFO[blob_id]['OPENCV']['status'].append(NOT_SET)
                    
            else:
                print('over camera_log_index:', camera_params_pos)

        if AUTO_LOOP == 1:
            frame_cnt += 1
        if SERVER == 0:
            cv2.imshow("Tracking", draw_frame)
        key = cv2.waitKey(1)
        # Exit if ESC key is
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == 27:
            print('ESC pressed')
            cv2.destroyAllWindows()
            sys.exit()
            return ERROR
        elif key & 0xFF == ord('c'):
            while True:
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
        elif key == ord('n'):
            frame_cnt += 1
            bboxes.clear()
            print('go next IMAGE')
        elif key == ord('b'):
            frame_cnt -= 1
            bboxes.clear()
            print('go back IMAGE')
    cv2.destroyAllWindows()

    data = OrderedDict()
    data['BLOB_INFO'] = BLOB_INFO
    pickle_data(WRITE, 'BLOB_INFO.pickle', data)
    data = OrderedDict()
    data['CAMERA_INFO'] = CAMERA_INFO
    pickle_data(WRITE, 'CAMERA_INFO.pickle', data)   
def remake_3d_for_blob_info(**kwargs):
    BLOB_CNT = kwargs.get('blob_cnt')
    info_name = kwargs.get('info_name')
    undistort = kwargs.get('undistort')
    opencv = kwargs.get('opencv')
    blender = kwargs.get('blender')
    ba_rt = kwargs.get('ba_rt')
    print('remake_3d_for_blob_info START')
    BLOB_INFO = pickle_data(READ, info_name, None)['BLOB_INFO']

    REMADE_3D_INFO_B = {}
    REMADE_3D_INFO_O = {}
    REMADE_3D_INFO_BA = {} # For BA_RT
    # Create a pretty printer
    pp = pprint.PrettyPrinter(indent=4)
    
    if opencv == DONE:
        print('#################### DYNAMIC RT (PNP SOLVER)  ####################')
        for blob_id in range(BLOB_CNT):   
            CNT = len(BLOB_INFO[blob_id]['points2D_D']['greysum'])
            print(f"BLOB_ID: {blob_id}, CNT {CNT}")
            REMADE_3D_INFO_O[blob_id] = []

            # PnP solver는 최소 3개 (P3P 사용할 경우) 4개 이상 필요함
            rt_first_O = NOT_SET
            points2D_D_first = NOT_SET
            points2D_U_first = NOT_SET
            # print(BLOB_INFO[blob_id]['OPENCV']['status'])
            for i in range(0, CNT):
                if BLOB_INFO[blob_id]['OPENCV']['status'][i] != DONE:
                    continue
                
                if rt_first_O == NOT_SET:
                    rt_first_O = {
                        'rvec': BLOB_INFO[blob_id]['OPENCV']['rt']['rvec'][i],
                        'tvec': BLOB_INFO[blob_id]['OPENCV']['rt']['tvec'][i]
                    } 
                    points2D_D_first = [BLOB_INFO[blob_id]['points2D_D']['greysum'][i]]
                    points2D_U_first = [BLOB_INFO[blob_id]['points2D_U']['greysum'][i]]
                else:
                    # Get the 2D coordinates for the first and current frame
                    points2D_D_current = [BLOB_INFO[blob_id]['points2D_D']['greysum'][i]]
                    points2D_U_current = [BLOB_INFO[blob_id]['points2D_U']['greysum'][i]]
                
                    rt_current_O = {
                        'rvec': BLOB_INFO[blob_id]['OPENCV']['rt']['rvec'][i],
                        'tvec': BLOB_INFO[blob_id]['OPENCV']['rt']['tvec'][i]
                    }
                    if undistort == 0:
                        remake_3d_O = remake_3d_point(camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][0],
                                                    rt_first_O, rt_current_O,
                                                    points2D_D_first, points2D_D_current).reshape(-1, 3)
                    else:
                        remake_3d_O = remake_3d_point(default_cameraK, default_cameraK,
                                                    rt_first_O, rt_current_O,
                                                    points2D_U_first, points2D_U_current).reshape(-1, 3)
                    REMADE_3D_INFO_O[blob_id].append(remake_3d_O.reshape(-1, 3))

    if blender == DONE:
        print('#################### STATIC RT (BLENDER)  ####################')
        # Sliding Window
        # def sliding_window(data, window_size):
        #     for i in range(len(data) - window_size + 1):
        #         yield data[i:i + window_size]
        # for blob_id in range(BLOB_CNT):
        #     CNT = len(BLOB_INFO[blob_id]['points2D_D']['greysum'])
        #     if CNT != 0:
        #         window_size = 10
        #         REMADE_3D_INFO_B[blob_id] = []
        #         for idx in sliding_window(range(CNT), window_size):
        #             # print(idx)
        #             rt_first_B = {
        #                 'rvec': BLOB_INFO[blob_id]['BLENDER']['rt']['rvec'][idx[0]],
        #                 'tvec': BLOB_INFO[blob_id]['BLENDER']['rt']['tvec'][idx[0]]
        #             }
        #             points2D_D_first = [BLOB_INFO[blob_id]['points2D_D']['greysum'][idx[0]]]
        #             points2D_U_first = [BLOB_INFO[blob_id]['points2D_U']['greysum'][idx[0]]]
        #             # Get the 2D coordinates for the first and current frame
        #             points2D_D_current = [BLOB_INFO[blob_id]['points2D_D']['greysum'][idx[1]]]
        #             points2D_U_current = [BLOB_INFO[blob_id]['points2D_U']['greysum'][idx[1]]]
        #             rt_current_B = {
        #                 'rvec': BLOB_INFO[blob_id]['BLENDER']['rt']['rvec'][idx[1]],
        #                 'tvec': BLOB_INFO[blob_id]['BLENDER']['rt']['tvec'][idx[1]]
        #             }
        #             if undistort == 0:
        #                 remake_3d_B = remake_3d_point(camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][0],
        #                                             rt_first_B, rt_current_B,
        #                                             points2D_D_first, points2D_D_current).reshape(-1, 3)
        #             else:
        #                 remake_3d_B = remake_3d_point(default_cameraK, default_cameraK,
        #                                             rt_first_B, rt_current_B,
        #                                             points2D_U_first, points2D_U_current).reshape(-1, 3)
        #             REMADE_3D_INFO_B[blob_id].append(remake_3d_B.reshape(-1, 3))




        # # 서로 다른 2개 조합
        # for blob_id in range(BLOB_CNT):
        #     CNT = len(BLOB_INFO[blob_id]['points2D_D']['greysum'])
        #     if CNT != 0:
        #         REMADE_3D_INFO_B[blob_id] = []
        #         for idx in combinations(range(CNT), 2):
        #             rt_first_B = {
        #                 'rvec': BLOB_INFO[blob_id]['BLENDER']['rt']['rvec'][idx[0]],
        #                 'tvec': BLOB_INFO[blob_id]['BLENDER']['rt']['tvec'][idx[0]]
        #             }
        #             points2D_D_first = [BLOB_INFO[blob_id]['points2D_D']['greysum'][idx[0]]]
        #             points2D_U_first = [BLOB_INFO[blob_id]['points2D_U']['greysum'][idx[0]]]
        #             # Get the 2D coordinates for the first and current frame
        #             points2D_D_current = [BLOB_INFO[blob_id]['points2D_D']['greysum'][idx[1]]]
        #             points2D_U_current = [BLOB_INFO[blob_id]['points2D_U']['greysum'][idx[1]]]
        #             rt_current_B = {
        #                 'rvec': BLOB_INFO[blob_id]['BLENDER']['rt']['rvec'][idx[1]],
        #                 'tvec': BLOB_INFO[blob_id]['BLENDER']['rt']['tvec'][idx[1]]
        #             }
        #             if undistort == 0:
        #                 remake_3d_B = remake_3d_point(camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][0],
        #                                             rt_first_B, rt_current_B,
        #                                             points2D_D_first, points2D_D_current).reshape(-1, 3)
        #             else:
        #                 remake_3d_B = remake_3d_point(default_cameraK, default_cameraK,
        #                                             rt_first_B, rt_current_B,
        #                                             points2D_U_first, points2D_U_current).reshape(-1, 3)
        #             REMADE_3D_INFO_B[blob_id].append(remake_3d_B.reshape(-1, 3))

        # 0 번 부터 순서대로 복원
        for blob_id in range(BLOB_CNT):
            CNT = len(BLOB_INFO[blob_id]['points2D_D']['greysum'])
            print(f"BLOB_ID: {blob_id}, CNT {CNT}")
            REMADE_3D_INFO_B[blob_id] = []

            # PnP solver는 최소 3개 (P3P 사용할 경우) 4개 이상 필요함
            rt_first_B = NOT_SET
            points2D_D_first = NOT_SET
            points2D_U_first = NOT_SET
            for i in range(0, CNT):
                if BLOB_INFO[blob_id]['BLENDER']['status'][i] != DONE:
                    continue
                
                if rt_first_B == NOT_SET:
                    rt_first_B = {
                        'rvec': BLOB_INFO[blob_id]['BLENDER']['rt']['rvec'][i],
                        'tvec': BLOB_INFO[blob_id]['BLENDER']['rt']['tvec'][i]
                    }
                    points2D_D_first = [BLOB_INFO[blob_id]['points2D_D']['greysum'][i]]
                    points2D_U_first = [BLOB_INFO[blob_id]['points2D_U']['greysum'][i]]
                else:
                    # Get the 2D coordinates for the first and current frame
                    points2D_D_current = [BLOB_INFO[blob_id]['points2D_D']['greysum'][i]]
                    points2D_U_current = [BLOB_INFO[blob_id]['points2D_U']['greysum'][i]]
                
                    rt_current_B = {
                        'rvec': BLOB_INFO[blob_id]['BLENDER']['rt']['rvec'][i],
                        'tvec': BLOB_INFO[blob_id]['BLENDER']['rt']['tvec'][i]
                    }
                    if undistort == 0:
                        remake_3d_B = remake_3d_point(camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][0],
                                                    rt_first_B, rt_current_B,
                                                    points2D_D_first, points2D_D_current).reshape(-1, 3)
                    else:
                        remake_3d_B = remake_3d_point(default_cameraK, default_cameraK,
                                                    rt_first_B, rt_current_B,
                                                    points2D_U_first, points2D_U_current).reshape(-1, 3)
                    REMADE_3D_INFO_B[blob_id].append(remake_3d_B.reshape(-1, 3))

               
    if ba_rt == DONE:
        # Iterate over the blob_ids in BLOB_INFO
        print('#################### DYNAMIC RT (BA RT)  ####################')
        for blob_id in range(BLOB_CNT):
            CNT = len(BLOB_INFO[blob_id]['points2D_D']['greysum'])
            print(f"BLOB_ID: {blob_id}, CNT {CNT}")
            REMADE_3D_INFO_BA[blob_id] = []

            # PnP solver는 최소 3개 (P3P 사용할 경우) 4개 이상 필요함
            rt_first_BA = NOT_SET
            points2D_D_first = NOT_SET
            points2D_U_first = NOT_SET
            for i in range(0, CNT):
                if BLOB_INFO[blob_id]['BA_RT']['status'][i] != DONE:
                    continue
                
                if rt_first_BA == NOT_SET:
                    rt_first_BA = {
                        'rvec': BLOB_INFO[blob_id]['BA_RT']['rt']['rvec'][i],
                        'tvec': BLOB_INFO[blob_id]['BA_RT']['rt']['tvec'][i]
                    }
                    points2D_D_first = [BLOB_INFO[blob_id]['points2D_D']['greysum'][i]]
                    points2D_U_first = [BLOB_INFO[blob_id]['points2D_U']['greysum'][i]]
                else:
                    # Get the 2D coordinates for the first and current frame
                    points2D_D_current = [BLOB_INFO[blob_id]['points2D_D']['greysum'][i]]
                    points2D_U_current = [BLOB_INFO[blob_id]['points2D_U']['greysum'][i]]
                
                    rt_current_BA = {
                        'rvec': BLOB_INFO[blob_id]['BA_RT']['rt']['rvec'][i],
                        'tvec': BLOB_INFO[blob_id]['BA_RT']['rt']['tvec'][i]
                    }
                    # Create the 3D points using the BA_RT data
                    if undistort == 0:
                        remake_3d_BA = remake_3d_point(camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][0],
                                                        rt_first_BA, rt_current_BA,
                                                        points2D_D_first, points2D_D_current).reshape(-1, 3)
                    else:
                        remake_3d_BA = remake_3d_point(default_cameraK, default_cameraK,
                                                        rt_first_BA, rt_current_BA,
                                                        points2D_U_first, points2D_U_current).reshape(-1, 3)
                    REMADE_3D_INFO_BA[blob_id].append(remake_3d_BA.reshape(-1, 3))
 

    file = 'REMADE_3D_INFO.pickle'
    data = OrderedDict()
    data['REMADE_3D_INFO_B'] = REMADE_3D_INFO_B
    data['REMADE_3D_INFO_O'] = REMADE_3D_INFO_O
    data['REMADE_3D_INFO_BA'] = REMADE_3D_INFO_BA # For BA_RT
    ret = pickle_data(WRITE, file, data)
    if ret != ERROR:
        print('data saved remake 3d')
def LSM(TARGET_DEVICE, MODEL_DATA):
    print('draw_result START')
    REMADE_3D_INFO_B = pickle_data(READ, 'REMADE_3D_INFO.pickle', None)['REMADE_3D_INFO_B']
    REMADE_3D_INFO_O = pickle_data(READ, 'REMADE_3D_INFO.pickle', None)['REMADE_3D_INFO_O']
    REMADE_3D_INFO_BA = pickle_data(READ, 'REMADE_3D_INFO.pickle', None)['REMADE_3D_INFO_BA']
    BA_3D = pickle_data(READ, 'BA_3D.pickle', None)['BA_3D']
    LED_INDICES = pickle_data(READ, 'BA_3D.pickle', None)['LED_INDICES']
    origin_pts = np.array(MODEL_DATA).reshape(-1, 3)

    # # BA 3D 정보와 origin_pts 간의 유클리드 거리를 계산하고, ax2에 그리기
    # ba_3d = {}
    # for i, blob_id in enumerate(LED_INDICES):
    #     point_3d = BA_3D[i].reshape(-1)

    #     if blob_id not in ba_3d:
    #         ba_3d[blob_id] = []  # 이 blob_id에 대한 리스트가 아직 없다면 새로 생성합니다.
    #     ba_3d[blob_id].append(point_3d)


    ba_3d_dict = {}
    for blob_id, data_list in REMADE_3D_INFO_BA.items():
        for data in data_list:
            point_3d = data.reshape(-1)
            if blob_id not in ba_3d_dict:
                ba_3d_dict[blob_id] = []  # 이 blob_id에 대한 리스트가 아직 없다면 새로 생성합니다.
            ba_3d_dict[blob_id].append(point_3d)
 
    # 각 blob_id에 대해 PCA를 적용하고, 첫 번째 주성분에 대한 중심을 계산합니다.
    centers_ba = {}
    for blob_id, points_3d in ba_3d_dict.items():
        pca = PCA(n_components=3)  # 3차원 PCA를 계산합니다.
        pca.fit(points_3d)
        # PCA의 첫 번째 주성분의 중심을 계산합니다.
        center = pca.mean_
        centers_ba[blob_id] = center  # 이후에는 center를 원하는대로 사용하면 됩니다


    # centers_ba에는 각 blob_id의 대표값이 저장되어 있습니다.
    print('\n')
    print('#################### PCA  ####################')
    PCA_ARRAY = []
    for blob_id, center in centers_ba.items():
        print(f"Center of PCA for blob_id {blob_id}: {center}")
        PCA_ARRAY.append(center)

    if TARGET_DEVICE == 'SEMI_SLAM_PLANE':
        PCA_ARRAY_LSM = module_lsm_2D(MODEL_DATA, PCA_ARRAY)
    else:
        PCA_ARRAY_LSM = module_lsm_3D(MODEL_DATA, PCA_ARRAY)
    PCA_ARRAY_LSM = [[round(x, 8) for x in sublist] for sublist in PCA_ARRAY_LSM]
    print('PCA_ARRAY_LSM\n')
    for blob_id, points_3d in enumerate(PCA_ARRAY_LSM):
        print(f"{points_3d},")   
        
    print('\n')
    print('#################### IQR_BA_RT  ####################')
    IQR_ARRAY = []
    for blob_id, points_3d in ba_3d_dict.items():
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


    if TARGET_DEVICE ==  'SEMI_SLAM_PLANE':
        IQR_ARRAY_LSM = module_lsm_2D(MODEL_DATA, IQR_ARRAY)
    else:
        IQR_ARRAY_LSM = module_lsm_3D(MODEL_DATA, IQR_ARRAY)
    IQR_ARRAY_LSM = [[round(x, 8) for x in sublist] for sublist in IQR_ARRAY_LSM]
    print('IQR_ARRAY_LSM(BA_RT)\n')
    for blob_id, points_3d in enumerate(IQR_ARRAY_LSM):
        print(f"{points_3d},")    
    

    # print('\n')
    # print('#################### IQR BA_3D  ####################')
    # IQR_ARRAY = []
    # for blob_id, points_3d in ba_3d.items():
    #     acc_blobs = points_3d.copy()
    #     acc_blobs_length = len(acc_blobs)
    #     if acc_blobs_length == 0:
    #         print('acc_blobs_length is 0 ERROR')
    #         continue

    #     remove_index_array = []
    #     med_blobs = [[], [], []]
    #     for blobs in acc_blobs:
    #         med_blobs[0].append(blobs[0])
    #         med_blobs[1].append(blobs[1])
    #         med_blobs[2].append(blobs[2])
            
    #     detect_outliers(med_blobs, remove_index_array)

    #     count = 0
    #     for index in remove_index_array:
    #         med_blobs[0].pop(index - count)
    #         med_blobs[1].pop(index - count)
    #         med_blobs[2].pop(index - count)
    #         count += 1

    #     mean_med_x = round(np.mean(med_blobs[0]), 8)
    #     mean_med_y = round(np.mean(med_blobs[1]), 8)
    #     mean_med_z = round(np.mean(med_blobs[2]), 8)
    #     IQR_ARRAY.append([mean_med_x, mean_med_y, mean_med_z])
    #     print(f"mean_med of IQR for blob_id {blob_id}: [{mean_med_x} {mean_med_y} {mean_med_z}]")       

    
    # if TARGET_DEVICE ==  'SEMI_SLAM_PLANE':
    #     IQR_ARRAY_LSM_BA_3D = module_lsm_2D(MODEL_DATA, IQR_ARRAY)
    # else:
    #     IQR_ARRAY_LSM_BA_3D = module_lsm_3D(MODEL_DATA, IQR_ARRAY)
    # IQR_ARRAY_LSM_BA_3D = [[round(x, 8) for x in sublist] for sublist in IQR_ARRAY_LSM_BA_3D]
    # print('IQR_ARRAY_LSM_BA_3D(BA_3D)\n')
    # for blob_id, points_3d in enumerate(IQR_ARRAY_LSM_BA_3D):
    #     print(f"{points_3d},")    



    file = 'RIGID_3D_TRANSFORM.pickle'
    data = OrderedDict()
    data['PCA_ARRAY_LSM'] = PCA_ARRAY_LSM
    data['IQR_ARRAY_LSM'] = IQR_ARRAY_LSM
    # data['IQR_ARRAY_LSM_BA_3D'] = IQR_ARRAY_LSM_BA_3D
    ret = pickle_data(WRITE, file, data)
    if ret != ERROR:
        print('data saved')
 
def BA_RT(**kwargs):
    info_name = kwargs.get('info_name')
    save_to = kwargs.get('save_to')
    target = kwargs.get('target')
    print('BA_RT START')
    CAMERA_INFO = pickle_data(READ, info_name, None)['CAMERA_INFO']   
    camera_indices = []
    point_indices = []
    estimated_RTs = []
    POINTS_2D = []
    POINTS_3D = []
    n_points = 0
    cam_id = 0

    if len(CAMERA_INFO) <= 0:
        return ERROR

    for frame_cnt, cam_info in CAMERA_INFO.items():
        # print(cam_info['LED_NUMBER'])

        points3D = cam_info['points3D']
        # 여기 다시 확인 해야 함
        rvec = cam_info[target]['rt']['rvec']
        tvec = cam_info[target]['rt']['tvec']
        points2D_D = cam_info['points2D']['greysum']
        points2D_U = cam_info['points2D_U']['greysum']
        # print('frame_cnt ', frame_cnt)
        # print('rvec ', rvec)
        # print('tvec ', tvec)   
        
        # Add camera parameters (rvec and tvec)
        estimated_RTs.append((rvec.ravel(), tvec.ravel()))

        # Adding 2D points
        POINTS_2D.extend(points2D_D if undistort == 0 else points2D_U)
        
        # Adding 3D points
        POINTS_3D.extend(points3D)
        
        # Adding indices
        camera_indices.extend([cam_id]*len(points3D))
        point_indices.extend(list(range(n_points, n_points+len(points3D))))

        n_points += len(points3D)
        cam_id += 1

    def fun(params, n_cameras, camera_indices, point_indices, points_2d, points_3d, camera_matrix):
        camera_params = params.reshape((n_cameras, 6))
        points_proj = []

        for i, POINT_3D in enumerate(points_3d[point_indices]):
            camera_index = camera_indices[i]
            rvec = camera_params[camera_index, :3]
            tvec = camera_params[camera_index, 3:]
            POINT_2D_PROJ, _ = cv2.projectPoints(POINT_3D,
                                                 np.array(rvec),
                                                 np.array(tvec),
                                                 camera_matrix[CAM_ID][0] if undistort == 0 else default_cameraK,
                                                 camera_matrix[CAM_ID][1] if undistort == 0 else default_dist_coeffs)
            points_proj.append(POINT_2D_PROJ[0][0])

        points_proj = np.array(points_proj)
        return (points_proj - points_2d).ravel()


    def bundle_adjustment_sparsity(n_cameras, camera_indices):
        m = camera_indices.size * 2
        n = n_cameras * 6
        A = lil_matrix((m, n), dtype=int)
        i = np.arange(camera_indices.size)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1
        return A

    # Convert the lists to NumPy arrays
    n_cameras = len(estimated_RTs)
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    camera_params = np.array(estimated_RTs).reshape(-1, 6)
    POINTS_2D = np.array(POINTS_2D).reshape(-1, 2)
    POINTS_3D = np.array(POINTS_3D).reshape(-1, 3)

    x0 = camera_params.ravel()
    A = bundle_adjustment_sparsity(n_cameras, camera_indices)
    
    print('\n')
    print('#################### BA  ####################')
    print('n_points', n_points)
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-6, method='trf',
                        args=(n_cameras, camera_indices, point_indices, POINTS_2D, POINTS_3D, camera_matrix))

    # You are only optimizing camera parameters, so the result only contains camera parameters data
    n_cameras_params = res.x.reshape((n_cameras, 6))
    # print("Optimized camera parameters: ", n_cameras_params, ' ', len(n_cameras_params))
    file = save_to
    data = OrderedDict()
    data['BA_RT'] = n_cameras_params
    data['camera_indices'] = camera_indices
    ret = pickle_data(WRITE, file, data)
    if ret != ERROR:
        print('data saved')   
def BA_3D_POINT(**kwargs):
    print('BA_3D_POINT START')
    RT = kwargs.get('RT')
    BLOB_INFO = pickle_data(READ, 'BLOB_INFO.pickle', None)['BLOB_INFO']
    REMADE_3D_INFO_B = pickle_data(READ, 'REMADE_3D_INFO.pickle', None)['REMADE_3D_INFO_B']
    REMADE_3D_INFO_O = pickle_data(READ, 'REMADE_3D_INFO.pickle', None)['REMADE_3D_INFO_O']
    camera_indices = []
    point_indices = []
    estimated_RTs = []
    POINTS_2D = []
    POINTS_3D = []
    n_points = 0
    cam_id = 0

    # Iterating over each blob_id in BLOB_INFO and REMADE_3D_INFO_B
    LED_INDICES = []
    for blob_id, blob_info in BLOB_INFO.items():
        remade_3d_info = REMADE_3D_INFO_B[blob_id]
        for frame_id in range(1, len(blob_info[RT]['rt']['rvec'])):
            # Adding 2D points
            if undistort == 0:
                POINTS_2D.append(blob_info['points2D_D']['greysum'][frame_id])
            else:
                POINTS_2D.append(blob_info['points2D_U']['greysum'][frame_id])
            # Adding 3D points
            POINTS_3D.append(remade_3d_info[frame_id - 1])
            # Adding RTs
            rvec = blob_info[RT]['rt']['rvec'][frame_id]
            tvec = blob_info[RT]['rt']['tvec'][frame_id]
            estimated_RTs.append((rvec.ravel(), tvec.ravel()))

            # Adding camera id
            camera_indices.append(cam_id)
            # Adding point index
            point_indices.append(cam_id)
            LED_INDICES.append(blob_id)
            cam_id += 1
        n_points += (len(blob_info[RT]['rt']['rvec']) - 1)

    def fun(params, n_points, camera_indices, point_indices, points_2d, camera_params, camera_matrix):
        """Compute residuals.
        `params` contains 3-D coordinates.
        """
        points_3d = params.reshape((n_points, 3))

        points_proj = []
        for i, POINT_3D in enumerate(points_3d[point_indices]):
            camera_index = camera_indices[i]
            # print('points_3d', POINT_3D, ' ', camera_index, ' ', i)
            # print('R', np.array(camera_params[camera_indices][i][0]))
            # print('T', np.array(camera_params[camera_indices][i][1]))
            POINT_2D_PROJ, _ = cv2.projectPoints(POINT_3D,
                                                 np.array(camera_params[camera_indices][i][0]),
                                                 np.array(camera_params[camera_indices][i][1]),
                                                 camera_matrix[CAM_ID][0] if undistort == 0 else default_cameraK,
                                                 camera_matrix[CAM_ID][1] if undistort == 0 else default_dist_coeffs)
            points_proj.append(POINT_2D_PROJ[0][0])

        points_proj = np.array(points_proj)
        return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity(n_points, point_indices):
        m = point_indices.size * 2
        n = n_points * 3
        A = lil_matrix((m, n), dtype=int)
        i = np.arange(point_indices.size)
        for s in range(3):
            A[2 * i, point_indices * 3 + s] = 1
            A[2 * i + 1, point_indices * 3 + s] = 1
        return A

    # Convert the lists to NumPy arrays
    n_cameras = len(estimated_RTs)
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    camera_params = np.array(estimated_RTs)
    POINTS_2D = np.array(POINTS_2D).reshape(-1, 2)
    POINTS_3D = np.array(POINTS_3D).reshape(-1, 3)

    # print('camera_params\n', camera_params.reshape(-1, 6))
    x0 = POINTS_3D.ravel()
    A = bundle_adjustment_sparsity(n_points, point_indices)
    
    print('\n')
    print('#################### BA  ####################')
    print('n_points', n_points)
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-6, method='trf',
                        args=(n_points, camera_indices, point_indices, POINTS_2D, camera_params, camera_matrix))

    # You are only optimizing points, so the result only contains point data
    n_points_3d = res.x.reshape((n_points, 3))
    # print("Optimized 3D points: ", n_points_3d, ' ', len(n_points_3d))
    file = 'BA_3D.pickle'
    data = OrderedDict()
    data['BA_3D'] = n_points_3d
    data['LED_INDICES'] = LED_INDICES
    ret = pickle_data(WRITE, file, data)
    if ret != ERROR:
        print('data saved')
def Check_Calibration_data_combination(combination_cnt):
    print('Check_Calibration_data_combination START')
    CAMERA_INFO = pickle_data(READ, 'CAMERA_INFO.pickle', None)['CAMERA_INFO']       
    RIGID_3D_TRANSFORM_IQR = pickle_data(READ, 'RIGID_3D_TRANSFORM.pickle', None)['IQR_ARRAY_LSM']

    print('Calibration cadidates')
    for blob_id, points_3d in enumerate(RIGID_3D_TRANSFORM_IQR):
        print(f"{points_3d},")    
    
    def STD_Analysis(points3D_data, label, combination):
        print(f"dataset:{points3D_data} combination_cnt:{combination}")
        frame_counts = []
        rvec_std_arr = []
        tvec_std_arr = []
        reproj_err_rates = []
        error_cnt = 0
        fail_reason = []
        for frame_cnt, cam_data in CAMERA_INFO.items():           
            
            rvec_list = []
            tvec_list = []
            reproj_err_list = []

            LED_NUMBER = cam_data['LED_NUMBER']
            points3D = cam_data[points3D_data]
            points2D = cam_data['points2D']['greysum']
            points2D_U = cam_data['points2D_U']['greysum']
            # print('frame_cnt: ',frame_cnt)
            # print('points2D_U: ',points2D_U)
            # print('points3D\n',points3D)
            LENGTH = len(LED_NUMBER)

            if LENGTH >= combination:
                for comb in combinations(range(LENGTH), combination):
                    for perm in permutations(comb):
                        points3D_perm = points3D[list(perm), :]
                        points2D_perm = points2D[list(perm), :]
                        points2D_U_perm = points2D_U[list(perm), :]
                        if combination >= 5:
                            METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC
                        elif combination == 4:
                            METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P

                        INPUT_ARRAY = [
                            CAM_ID,
                            points3D_perm,
                            points2D_perm if undistort == 0 else points2D_U_perm,
                            camera_matrix[CAM_ID][0] if undistort == 0 else default_cameraK,
                            camera_matrix[CAM_ID][1] if undistort == 0 else default_dist_coeffs
                        ]
                        _, rvec, tvec, _ = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
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

                        RER = reprojection_error(points3D_perm,
                                                points2D_perm,
                                                rvec, tvec,
                                                camera_matrix[CAM_ID][0],
                                                camera_matrix[CAM_ID][1])
                        if RER > 10.0:
                            error_cnt+=1
                            # print('points3D_perm ', points3D_perm)
                            # print('points3D_data ', points3D_data)
                            # print('list(perm): ', list(perm), ' RER: ', RER)
                            fail_reason.append([points3D_data, error_cnt, list(perm), RER, rvec.flatten(), tvec.flatten()])

    
                        rvec_list.append(np.linalg.norm(rvec))
                        tvec_list.append(np.linalg.norm(tvec))                      
                        reproj_err_list.append(RER)                                
                
                if rvec_list and tvec_list and reproj_err_list:
                    frame_counts.append(frame_cnt)
                    rvec_std = np.std(rvec_list)
                    tvec_std = np.std(tvec_list)
                    reproj_err_rate = np.mean(reproj_err_list)

                    rvec_std_arr.append(rvec_std)
                    tvec_std_arr.append(tvec_std)
                    reproj_err_rates.append(reproj_err_rate)


        return frame_counts, rvec_std_arr, tvec_std_arr, reproj_err_rates, label, fail_reason


    all_data = []
    for COMBINATION in combination_cnt:
        fig, axs = plt.subplots(3, 1, figsize=(15, 15))
        fig.suptitle(f'Calibration Data Analysis for Combination: {COMBINATION}')  # Set overall title

        points3D_datas = ['points3D', 'points3D_IQR']
        colors = ['r', 'b']
        summary_text = ""

        for idx, points3D_data in enumerate(points3D_datas):
            frame_counts, rvec_std_arr, tvec_std_arr, reproj_err_rates, label, fail_reason = STD_Analysis(points3D_data, points3D_data, COMBINATION)

            axs[0].plot(frame_counts, rvec_std_arr, colors[idx]+'-', label=f'rvec std {label}', alpha=0.5)
            axs[0].plot(frame_counts, tvec_std_arr, colors[idx]+'--', label=f'tvec std {label}', alpha=0.5)
            axs[1].plot(frame_counts, reproj_err_rates, colors[idx], label=f'Reprojection error rate {label}', alpha=0.5)
            
            # Calculate and store the average and standard deviation for each data set
            avg_rvec_std = np.mean(rvec_std_arr)
            std_rvec_std = np.std(rvec_std_arr)
            avg_tvec_std = np.mean(tvec_std_arr)
            std_tvec_std = np.std(tvec_std_arr)
            avg_reproj_err = np.mean(reproj_err_rates)
            std_reproj_err = np.std(reproj_err_rates)
            
            summary_text += f"== {label} ==\n"
            summary_text += f"Rvec Std: Mean = {avg_rvec_std:.6f}, Std = {std_rvec_std:.6f}\n"
            summary_text += f"Tvec Std: Mean = {avg_tvec_std:.6f}, Std = {std_tvec_std:.6f}\n"
            summary_text += f"Reproj Err: Mean = {avg_reproj_err:.6f}, Std = {std_reproj_err:.6f}\n"
            # summary_text += f"error cnt (over RER 2px) {error_cnt:.6f}\n"
            summary_text += "\n"
            all_data.append([label, COMBINATION, avg_rvec_std, std_rvec_std, avg_tvec_std, std_tvec_std, avg_reproj_err, std_reproj_err])  # Store data for all combinations

        axs[0].legend()
        axs[0].set_xlabel('frame_cnt')
        axs[0].set_ylabel('std')
        axs[0].set_title('Standard Deviation of rvec and tvec Magnitude per Frame')

        axs[1].legend()
        axs[1].set_xlabel('frame_cnt')
        axs[1].set_ylabel('error rate')
        axs[1].set_title('Mean Reprojection Error Rate per Frame')
        
        axs[2].axis('off')  # Hide axes for the text plot
        axs[2].text(0, 0, summary_text, fontsize=10)

        # Reducing the number of X-ticks to avoid crowding
        for ax in axs[:2]:
            ax.set_xticks(ax.get_xticks()[::5])

        plt.subplots_adjust(hspace=0.5)  # Add space between subplots
    original_labels = [f"{item[0]} {item[1]}" for item in all_data] # Here, we first create original labels
    labels = combination_cnt * (len(original_labels) // len(combination_cnt)) # Now we can use len(original_labels)

    avg_rvec_std_values = [item[2] for item in all_data]
    std_rvec_std_values = [item[3] for item in all_data]
    avg_tvec_std_values = [item[4] for item in all_data]
    std_tvec_std_values = [item[5] for item in all_data]
    avg_reproj_err_values = [item[6] for item in all_data]
    std_reproj_err_values = [item[7] for item in all_data]

    x = np.arange(len(labels) // 2)
    print(x)
    print(labels)
    print(labels[:len(combination_cnt)])
    width = 0.35
    fig, axs = plt.subplots(4, 1, figsize=(15, 30)) # increase the figure size

    # Rvec subplot
    rects1 = axs[0].bar(x - width / 4, avg_rvec_std_values[::2], width / 2, color='r', label='Avg Rvec Std for points3D')
    rects2 = axs[0].bar(x + width / 4, avg_rvec_std_values[1::2], width / 2, color='b', label='Avg Rvec Std for points3D_IQR')

    axs[0].set_xlabel('Combination')
    axs[0].set_ylabel('Values')
    axs[0].set_title('Average Rvec Standard Deviations for All Combinations')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(labels[:len(combination_cnt)])
    axs[0].legend()

    # Tvec subplot
    rects3 = axs[1].bar(x - width / 4, avg_tvec_std_values[::2], width / 2, color='r', label='Avg Tvec Std for points3D')
    rects4 = axs[1].bar(x + width / 4, avg_tvec_std_values[1::2], width / 2, color='b', label='Avg Tvec Std for points3D_IQR')

    axs[1].set_xlabel('Combination')
    axs[1].set_ylabel('Values')
    axs[1].set_title('Average Tvec Standard Deviations for All Combinations')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(labels[:len(combination_cnt)])
    axs[1].legend()

    # Reproj_err subplot
    rects5 = axs[2].bar(x - width / 4, avg_reproj_err_values[::2], width / 2, color='r', label='Avg Reproj Err for points3D')
    rects6 = axs[2].bar(x + width / 4, avg_reproj_err_values[1::2], width / 2, color='b', label='Avg Reproj Err for points3D_IQR')

    axs[2].set_xlabel('Combination')
    axs[2].set_ylabel('Values')
    axs[2].set_title('Average Reprojection Error for All Combinations')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(labels[:len(combination_cnt)])
    axs[2].legend()

      # Organize data into a dictionary with column labels as keys
    summary_data = {
        'LEDCount:': labels[:len(combination_cnt)],
        'AvgRvecStd(points3D):': avg_rvec_std_values[::2],
        'AvgRvecStd(points3D_IQR):': avg_rvec_std_values[1::2],
        'AvgTvecStd(points3D):': avg_tvec_std_values[::2],
        'AvgTvecStd(points3D_IQR):': avg_tvec_std_values[1::2],
        'AvgReprojErr(points3D):': avg_reproj_err_values[::2],
        'AvgReprojErr(points3D_IQR):': avg_reproj_err_values[1::2]
    }

    # Create DataFrame from dictionary
    df = pd.DataFrame(summary_data)
    # 'LEDCount:' 열을 제외한 새로운 DataFrame 생성
    df_without_ledcount = df.drop('LEDCount:', axis=1)

    # 통계를 가져옵니다
    desc_stats = df_without_ledcount.describe()

    # # Get descriptive statistics
    # desc_stats = df.describe()

    # Convert to string and add to summary text
    summary_text = df.to_string(index=False)
    summary_text += "\n\nDescriptive Statistics:\n" + desc_stats.to_string()

    # Remove existing text subplot
    fig.delaxes(axs[3])

    # Add table as text subplot
    axs[3] = fig.add_subplot(414)
    axs[3].axis('off')
    axs[3].text(0.5, 0.5, summary_text, ha='center', va='center')

    # Print to console
    print(summary_text)

    # Write to log file
    with open('summary.log', 'w') as f:
        f.write(summary_text)

    file = 'FAIL_REASON.pickle'
    data = OrderedDict()
    data['FAIL_REASON'] = fail_reason
    ret = pickle_data(WRITE, file, data)
    if ret != ERROR:
        print('data saved')
    
    plt.subplots_adjust(hspace=0.5)

def init_plot(MODEL_DATA):
    root = tk.Tk()
    width_px = root.winfo_screenwidth()
    height_px = root.winfo_screenheight()

    # 모니터 해상도에 맞게 조절
    mpl.rcParams['figure.dpi'] = 120  # DPI 설정
    monitor_width_inches = width_px / mpl.rcParams['figure.dpi']  # 모니터 너비를 인치 단위로 변환
    monitor_height_inches = height_px / mpl.rcParams['figure.dpi']  # 모니터 높이를 인치 단위로 변환

    fig = plt.figure(figsize=(monitor_width_inches, monitor_height_inches), num='Camera Simulator')

    plt.rcParams.update({'font.size': 7})
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
    ax1 = plt.subplot(gs[0], projection='3d')
    ax2 = plt.subplot(gs[1])

    led_number = len(MODEL_DATA)
    ax2.set_title('distance')
    ax2.set_xlim([0, led_number])  # x축을 LED 번호의 수 만큼 설정합니다. 이 경우 14개로 설정됩니다.
    ax2.set_xticks(range(led_number))  # x축에 표시되는 눈금을 LED 번호의 수 만큼 설정합니다.
    ax2.set_xticklabels(range(led_number))  # x축에 표시되는 눈금 라벨을 LED 번호의 수 만큼 설정합니다.

    origin_pts = np.array(MODEL_DATA).reshape(-1, 3)
    ax1.set_title('3D plot')    
    ax1.scatter(origin_pts[:, 0], origin_pts[:, 1], origin_pts[:, 2], color='gray', alpha=1.0, marker='o', s=10, label='ORIGIN')
    
    ax1.scatter(0, 0, 0, marker='o', color='k', s=20)
    ax1.set_xlim([-0.2, 0.2])
    ax1.set_xlabel('X')
    ax1.set_ylim([-0.2, 0.2])
    ax1.set_ylabel('Y')
    ax1.set_zlim([-0.2, 0.2])
    ax1.set_zlabel('Z')
    scale = 1.5
    f = zoom_factory(ax1, base_scale=scale)
    
    return ax1, ax2
def init_camera_path(script_dir, video_path, first_image_path):
    bboxes = []
    centers1 = []
    json_file = os.path.join(script_dir, './init_blob_area.json')
    json_data = rw_json_data(READ, json_file, None)
    if json_data != ERROR:
        bboxes = json_data['bboxes']
    CAPTURE_DONE = NOT_SET
    while  True:
        frame_0 = cv2.imread(first_image_path)
        draw_frame = frame_0.copy()
        _, frame_0 = cv2.threshold(cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                    cv2.THRESH_TOZERO)

        cv2.putText(draw_frame, f"[{first_image_path}]", (draw_frame.shape[1] - 400, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        height, width = frame_0.shape
        center_x, center_y = width // 2, height // 2
        cv2.line(draw_frame, (0, center_y), (width, center_y), (255, 255, 255), 1)
        cv2.line(draw_frame, (center_x, 0), (center_x, height), (255, 255, 255), 1)
                
        blob_area = detect_led_lights(frame_0, TRACKER_PADDING, 5, 500)
        cv2.namedWindow('image')
        partial_click_event = functools.partial(click_event, frame=frame_0, blob_area_0=blob_area, bboxes=bboxes)
        cv2.setMouseCallback('image', partial_click_event)
        key = cv2.waitKey(1)

        if key == ord('c'):
            print('clear area')
            bboxes.clear()
        elif key == ord('s'):
            print('save blob area')
            json_data = OrderedDict()
            json_data['bboxes'] = bboxes
            # Write json data
            rw_json_data(WRITE, json_file, json_data)
        elif key & 0xFF == 27:
            print('ESC pressed')
            cv2.destroyAllWindows()
            return
        elif key == ord('q'):
            print('go next step')
            break
        elif key == ord('n'):
            bboxes.clear()
            print('go next IMAGE')
        elif key == ord('b'):
            bboxes.clear()
            print('go back IMAGE')
        elif key == ord('p'):
            print('calculation data')
            print('bboxes', bboxes)

            LED_NUMBERS = []
            points2D_D = []
            points2D_U = []
            points3D = []
            for AREA_DATA in bboxes:
                IDX = int(AREA_DATA['idx'])
                bbox = AREA_DATA['bbox']
                (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                gcx, gcy, gsize = find_center(frame_0, (x, y, w, h))
                if gsize < BLOB_SIZE:
                    continue
                cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
                centers1.append((gcx, gcy, bbox))
                print(f"{IDX} : {gcx}, {gcy}")
                LED_NUMBERS.append(IDX)
                points3D.append(MODEL_DATA[IDX])
                temp_blobs = np.array([gcx, gcy], dtype=np.float64)
                points2D_D.append(temp_blobs)
                points2D_U.append(np.array(cv2.undistortPoints(temp_blobs, camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][1])).reshape(-1, 2))
            
 
            print('START Pose Estimation')
            points2D_D = np.array(np.array(points2D_D).reshape(len(points2D_D), -1), dtype=np.float64)
            points2D_U = np.array(np.array(points2D_U).reshape(len(points2D_U), -1), dtype=np.float64)
            points3D = np.array(points3D, dtype=np.float64)

            print('LED_NUMBERS: ', LED_NUMBERS)
            print('points2D_D\n', points2D_D)
            print('points2D_U\n', points2D_U)
            print('points3D\n', points3D)                 
            LENGTH = len(LED_NUMBERS)
            if LENGTH >= 4:
                print('PnP Solver OpenCV')
                if LENGTH >= 5:
                    METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC
                elif LENGTH == 4:
                    METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P
                INPUT_ARRAY = [
                    CAM_ID,
                    points3D,
                    points2D_D if undistort == 0 else points2D_U,
                    camera_matrix[CAM_ID][0] if undistort == 0 else default_cameraK,
                    camera_matrix[CAM_ID][1] if undistort == 0 else default_dist_coeffs
                ]
                _, rvec, tvec, _ = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
                print('PnP_Solver rvec:', rvec.flatten(), ' tvec:',  tvec.flatten())
            
            INIT_IMAGE = copy.deepcopy(CAMERA_INFO_STRUCTURE)
            INIT_IMAGE['points2D']['greysum'] = points2D_D
            INIT_IMAGE['points2D_U']['greysum'] = points2D_U            
            INIT_IMAGE['LED_NUMBER'] =LED_NUMBERS
            INIT_IMAGE['points3D'] =points3D
            CAPTURE_DONE = DONE
            break

        draw_blobs_and_ids(draw_frame, blob_area, bboxes)
        if SERVER == 0:
            cv2.imshow('image', draw_frame)

    cv2.destroyAllWindows()
    
    
    if CAPTURE_DONE == DONE:               
        cap = cv2.VideoCapture(video_path)

        # Get total number of frames
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        current_frame = 0

        print('total_frames: ', total_frames)

        S_closest_img = None
        S_min_distance = float('inf')
        E_closest_img = None
        E_min_distance = float('inf')
        while(cap.isOpened()):
            # Read frames
            ret, frame = cap.read()

            if ret:
                draw_frame = frame.copy()        
                _, frame = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                        cv2.THRESH_TOZERO)
                        

                cv2.putText(draw_frame, f'Frame {current_frame}/{total_frames}', 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0), 
                        1, 
                        cv2.LINE_AA)
                height, width = frame.shape
                center_x, center_y = width // 2, height // 2
                cv2.line(draw_frame, (0, center_y), (width, center_y), (255, 255, 255), 1)
                cv2.line(draw_frame, (center_x, 0), (center_x, height), (255, 255, 255), 1)            

                # LED blob 찾기
                blobs2 = detect_led_lights(frame, TRACKER_PADDING, 5, 500)
                # 두 번째 이미지의 LED blob 중심점 계산
                centers2 = []
                for blob_id, blob in enumerate(blobs2):
                    gcx, gcy, gsize = find_center(frame, blob)
                    if gsize < BLOB_SIZE:
                        continue
                    cv2.rectangle(draw_frame, (blob[0], blob[1]), (blob[0] + blob[2], blob[1] + blob[3]), (255, 255, 255), 1)
                    cv2.putText(draw_frame, f"{blob_id}", (blob[0], blob[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    centers2.append((gcx, gcy, blob))


                for center in centers1:
                    cv2.circle(draw_frame, (int(center[0]), int(center[1])), 1, (0, 255, 0), -1)

                for center in centers2:
                    cv2.circle(draw_frame, (int(center[0]), int(center[1])), 1, (0, 0, 255), -1)

                if current_frame < total_frames / 2:
                    if len(centers1) == len(centers2):
                        max_distance = max(np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                                        for c1, c2 in zip(centers1, centers2))            
                        if max_distance < S_min_distance:
                            S_min_distance = max_distance
                            S_closest_img = current_frame

                            # S_min_distance값을 업데이트 할 때 마다 이미지에 값을 업데이트
                            cv2.putText(draw_frame, f"S min distance: {S_min_distance}", 
                                        (50, 75), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        1, 
                                        (0, 255, 0), 
                                        1, 
                                        cv2.LINE_AA)
                else:
                    if len(centers1) == len(centers2):
                        max_distance = max(np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                                        for c1, c2 in zip(centers1, centers2))            
                        if max_distance < E_min_distance:
                            E_min_distance = max_distance
                            E_closest_img = current_frame

                            # E_min_distance값을 업데이트 할 때 마다 이미지에 값을 업데이트
                            cv2.putText(draw_frame, f"E min distance: {E_min_distance}", 
                                        (50, 100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        1, 
                                        (0, 255, 0), 
                                        1, 
                                        cv2.LINE_AA)


                # Show the frames
                if SERVER == 0:
                    cv2.imshow('Frame', draw_frame)

                # Wait for a key press and break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                current_frame += 1
            else:
                break

        print(f'The closest image START {S_closest_img} END {E_closest_img}')
        
        cap = cv2.VideoCapture(video_img_path)
        # Get the start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, S_closest_img)  # Frame indices start from 0
        ret1, frame_start = cap.read()

        # Get the closest_img frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, E_closest_img)  # Frame indices start from 0
        ret2, frame_closest = cap.read()

        if ret1 and ret2:  # If both frames are read correctly

            # Define range for white color in RGB
            lower_white = np.array([200, 200, 200])
            upper_white = np.array([255, 255, 255])

            # Threshold the RGB image to get only white colors
            mask_start = cv2.inRange(frame_start, lower_white, upper_white)
            mask_closest = cv2.inRange(frame_closest, lower_white, upper_white)

            # Change white (where mask is 1) to red and blue
            frame_start[mask_start == 255] = [0, 0, 255]
            frame_closest[mask_closest == 255] = [255, 0, 0]

            # Weighted addition of two frames
            overlapped_frame = cv2.addWeighted(frame_start, 0.5, frame_closest, 0.5, 0)
            if SERVER == 0:
                cv2.imshow('Overlapped frame', overlapped_frame)
            cv2.waitKey(0)  # Wait until any key is pressed

        cap.release()
        cv2.destroyAllWindows()
    
    return S_closest_img, E_closest_img
def save_camera_position(TARGET_DEVICE):
    CAMERA_INFO = pickle_data(READ, 'CAMERA_INFO.pickle', None)['CAMERA_INFO']    
    with open(f"camera_log_final_{TARGET_DEVICE}.txt", "w") as file:
        for key, camera_info in CAMERA_INFO.items():
            print('frame_cnt: ', key)
            barvec = camera_info['BA_RT']['rt']['rvec']
            batvec = camera_info['BA_RT']['rt']['tvec']
            print('barvec:', barvec.flatten())
            print('batvec:', batvec.flatten())

            # File writing
            rvec_str = ' '.join(map(str, barvec.flatten()))
            tvec_str = ' '.join(map(str, batvec.flatten()))
            file.write(f"Frame:{int(key) + 1}, Rvec:[{rvec_str}], Tvec:[{tvec_str}]\n")
        
        print('camera_log_final.txt saved')


def draw_result(MODEL_DATA, **kwargs):
    print('draw_result START')
    ax1 = kwargs.get('ax1')
    ax2 = kwargs.get('ax2')
    opencv = kwargs.get('opencv')
    blender = kwargs.get('blender')
    ba_rt = kwargs.get('ba_rt')
    ba_3d = kwargs.get('ba_3d')
    origin_pts = np.array(MODEL_DATA).reshape(-1, 3)

    if opencv == DONE:
        REMADE_3D_INFO_O = pickle_data(READ, 'REMADE_3D_INFO.pickle', None)['REMADE_3D_INFO_O']
        for blob_id, data_list in REMADE_3D_INFO_O.items():
            distances_remade = []
            for data in data_list:
                point_3d = data.reshape(-1)
                distance = np.linalg.norm(origin_pts[blob_id] - point_3d)
                distances_remade.append(distance)
                if distances_remade.index(distance) == 0:
                    ax1.scatter(point_3d[0], point_3d[1], point_3d[2], color='red', alpha=0.3, marker='o', s=7,
                                label='OPENCV')
                else:
                    ax1.scatter(point_3d[0], point_3d[1], point_3d[2], color='red', alpha=0.3, marker='o', s=7)
            ax2.scatter([blob_id] * len(distances_remade), distances_remade, color='red', alpha=0.5, marker='o', s=10,
                        label='OPENCV' if blob_id == list(REMADE_3D_INFO_O.keys())[0] else "_nolegend_")

    if blender == DONE:
        REMADE_3D_INFO_B = pickle_data(READ, 'REMADE_3D_INFO.pickle', None)['REMADE_3D_INFO_B']
        for blob_id, data_list in REMADE_3D_INFO_B.items():
            distances_remade = []
            for data in data_list:
                point_3d = data.reshape(-1)
                distance = np.linalg.norm(origin_pts[blob_id] - point_3d)
                distances_remade.append(distance)
                if distances_remade.index(distance) == 0:
                    ax1.scatter(point_3d[0], point_3d[1], point_3d[2], color='blue', alpha=0.3, marker='o', s=7,
                                label='BLENDER')
                else:
                    ax1.scatter(point_3d[0], point_3d[1], point_3d[2], color='blue', alpha=0.3, marker='o', s=7)
            ax2.scatter([blob_id] * len(distances_remade), distances_remade, color='blue', alpha=0.5, marker='o', s=10,
                        label='BLENDER' if blob_id == list(REMADE_3D_INFO_B.keys())[0] else "_nolegend_")

    if ba_rt == DONE:
        REMADE_3D_INFO_BA = pickle_data(READ, 'REMADE_3D_INFO.pickle', None)['REMADE_3D_INFO_BA']
        for blob_id, data_list in REMADE_3D_INFO_BA.items():
            distances_remade = []
            for data in data_list:
                point_3d = data.reshape(-1)
                distance = np.linalg.norm(origin_pts[blob_id] - point_3d)
                distances_remade.append(distance)
                if distances_remade.index(distance) == 0:
                    ax1.scatter(point_3d[0], point_3d[1], point_3d[2], color='magenta', alpha=0.3, marker='o', s=7,
                                label='BA_RT')
                else:
                    ax1.scatter(point_3d[0], point_3d[1], point_3d[2], color='magenta', alpha=0.3, marker='o', s=7)
            ax2.scatter([blob_id] * len(distances_remade), distances_remade, color='magenta', alpha=0.5, marker='o', s=10,
                        label='BA_RT' if blob_id == list(REMADE_3D_INFO_O.keys())[0] else "_nolegend_")
        
    if ba_3d == DONE:
        BA_3D = pickle_data(READ, 'BA_3D.pickle', None)['BA_3D']
        LED_INDICES = pickle_data(READ, 'BA_3D.pickle', None)['LED_INDICES']
        distances_ba = []
        for i, blob_id in enumerate(LED_INDICES):
            point_3d = BA_3D[i].reshape(-1)
            distance = np.linalg.norm(origin_pts[blob_id] - point_3d)
            distances_ba.append(distance)
            ax1.scatter(point_3d[0], point_3d[1], point_3d[2], color='green', alpha=0.3, marker='o', s=7,
                        label='BA_3D')
        ax2.scatter(LED_INDICES, distances_ba, color='green', alpha=0.5, marker='o', s=10, label='BA_3D')


        
    # Remove duplicate labels in legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    by_label1 = dict(zip(labels1, handles1))
    ax1.legend(by_label1.values(), by_label1.keys())

    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    ax2.legend(by_label2.values(), by_label2.keys())



    '''
    1. TRACKER
        Specify the tracker type 특정 버전 에서만 됨
        pip uninstall opencv-python
        pip uninstall opencv-contrib-python
        pip uninstall opencv-contrib-python-headless
        pip3 install opencv-contrib-python==4.5.5.62
    
    2. poselib
        linux system only
    '''


if __name__ == "__main__":

    SERVER = 0
    AUTO_LOOP = 1
    DO_P3P = 0
    DO_PYRAMID = 1
    SOLUTION = 1
    CV_MAX_THRESHOLD = 255
    CV_MIN_THRESHOLD = 150

    # Camera RT 마지막 버전 test_7
    TARGET_DEVICE = 'ARCTURAS'

    if TARGET_DEVICE == 'RIFTS':
        # Test_7 보고
        # 0, 2
        RIFTS_PATTERN_RIGHT = [0,0,1,0,1,0,1,0,1,0,1,0,1,0,0]
        LEDS_POSITION = RIFTS_PATTERN_RIGHT
        LEFT_RIGHT_DIRECTION = PLUS
        BLOB_SIZE = 100
        TOP_BOTTOM_LINE_Y = int(CAP_PROP_FRAME_HEIGHT / 2)
        controller_name = 'rifts_right_9'
        camera_log_path = f"./render_img/{controller_name}/test_7/camera_log_final.txt"
        camera_img_path = f"./render_img/{controller_name}/test_7/"
        combination_cnt = [4,5]
        MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/rifts_right_9.json"))
        START_FRAME = 0
        STOP_FRAME = 121
        THRESHOLD_DISTANCE = 10
        TRACKER_PADDING = 2
    elif TARGET_DEVICE == 'ARCTURAS':
        ARCTURAS_PATTERN_RIGHT = [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
        LEDS_POSITION = ARCTURAS_PATTERN_RIGHT
        LEFT_RIGHT_DIRECTION = MINUS
        BLOB_SIZE = 100
        TOP_BOTTOM_LINE_Y = int(CAP_PROP_FRAME_HEIGHT / 2)
        controller_name = 'arcturas'
        camera_log_path = f"./render_img/{controller_name}/test_2/camera_log_final_ARCTURAS.txt"
        camera_img_path = f"./render_img/{controller_name}/test_2/"
        combination_cnt = [4,5]
        MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/arcturas_right_1_self.json"))
        START_FRAME = 0
        STOP_FRAME = 121
        THRESHOLD_DISTANCE = 10
        TRACKER_PADDING = 2
    elif TARGET_DEVICE == 'SEMI_SLAM_CURVE':
        # Test_6 보고
        # 6, 7
        SEMI_SLAM_CURVE = [0,1,0,1,0,1,0,1]
        LEDS_POSITION = SEMI_SLAM_CURVE
        LEFT_RIGHT_DIRECTION = MINUS
        BLOB_SIZE = 150
        TOP_BOTTOM_LINE_Y = int(CAP_PROP_FRAME_HEIGHT / 2)
        controller_name = 'semi_slam_curve'
        camera_log_path = f"./render_img/{controller_name}/test_6/camera_log_final.txt"
        camera_img_path = f"./render_img/{controller_name}/test_6/"
        combination_cnt = [4,5]
        MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/semi_slam_curve.json"))
        START_FRAME = 0
        STOP_FRAME = 72
        THRESHOLD_DISTANCE = 10
        TRACKER_PADDING = 5
    elif TARGET_DEVICE == 'SEMI_SLAM_PLANE':
        # 6,5
        SEMI_SLAM_PLANE = [0,1,0,1,0,1,0]
        LEDS_POSITION = SEMI_SLAM_PLANE
        LEFT_RIGHT_DIRECTION = MINUS
        BLOB_SIZE = 300
        TOP_BOTTOM_LINE_Y = int(CAP_PROP_FRAME_HEIGHT / 2)
        controller_name = 'semi_slam_plane'
        camera_log_path = f"./render_img/{controller_name}/test_41/camera_log_final.txt"
        camera_img_path = f"./render_img/{controller_name}/test_41/"
        combination_cnt = [4,5]
        MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/semi_slam_plane.json"))
        START_FRAME = 30
        STOP_FRAME = 50
        THRESHOLD_DISTANCE = 10
        TRACKER_PADDING = 5 
    elif TARGET_DEVICE == 'SEMI_SLAM_POLYHEDRON':
        # 5,6 
        SEMI_SLAM_POLYHEDRON = [0,1,0,1,0,0,1]
        LEDS_POSITION = SEMI_SLAM_POLYHEDRON
        LEFT_RIGHT_DIRECTION = MINUS
        BLOB_SIZE = 300
        TOP_BOTTOM_LINE_Y = int(CAP_PROP_FRAME_HEIGHT / 2)
        controller_name = 'semi_slam_polyhedron'
        camera_log_path = f"./render_img/{controller_name}/test_21/camera_log_final.txt"
        camera_img_path = f"./render_img/{controller_name}/test_21/"
        combination_cnt = [4,5]
        MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/semi_slam_polyhedron.json"))
        START_FRAME = 25
        STOP_FRAME = 48
        THRESHOLD_DISTANCE = 10
        TRACKER_PADDING = 5
    else:
        # 피라미드 하면 이상해짐....
        print(f"DEVICE TEST {TARGET_DEVICE}")
        RIFTS_PATTERN_RIGHT = [0,0,1,0,1,0,1,0,1,0,1,0,1,0,0]
        LEDS_POSITION = RIFTS_PATTERN_RIGHT
        LEFT_RIGHT_DIRECTION = PLUS
        BLOB_SIZE = 250
        TOP_BOTTOM_LINE_Y = 494
        controller_name = 'rifts_right_9'
        camera_log_path = f"./tmp/render/RIFTS/{controller_name}/camera_log_final.txt"
        camera_img_path = f"./tmp/render/RIFTS/{controller_name}/"
        combination_cnt = [4,5]
        MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/rifts_right_9.json"))
        START_FRAME = 0
        STOP_FRAME = 121
        THRESHOLD_DISTANCE = 10
        TRACKER_PADDING = 2

    
    BLOB_CNT = len(MODEL_DATA)
    print('PTS')
    for i, leds in enumerate(MODEL_DATA):
        print(f"{np.array2string(leds, separator=', ')},")
    print('DIR')
    for i, dir in enumerate(DIRECTION):
        print(f"{np.array2string(dir, separator=', ')},")
    show_calibrate_data(np.array(MODEL_DATA), np.array(DIRECTION))
    # start, end = init_camera_path(script_dir, 'output_rifts_right_9.mkv', 'start_capture_rifts_right_9.jpg')

    ax1, ax2 = init_plot(MODEL_DATA)
    bboxes = blob_setting(script_dir, SERVER, f"{script_dir}/render_img/{controller_name}/blob_area.json")

    if SOLUTION == 1:
        ######################################## SOLUTION 1 ########################################
        # 설계 좌표가 있는 경우 BA_RT : 설계 값에  BLENDER RT를 BA 처리 할 수 있음
        # Remake 3D 값을 BA_RT 처리한 RT에 BA_3D_POINT 할 수 있음
        '''
        1차 보정
        * SEED PATH 사용 이유
            -설계값으로 초기 카메라 path를 만들 때 노이즈와 이상치를 fit circular algorithm으로 제거함
            -BA에 안정적인 input R|T 제공하여 보정
            -pnpSolver로만 보정하려고 하면 특정 카메라뷰에서는 블롭 갯수가 부족하거나 BA를 위한 데이터 모집단이 부족할 수 있음
            -블롭갯수가 부족한 위치를 원형방정식으로 구한 Blender RT로 채우고 
            해당 부분은 BA와 소수의 블롭으로 BA를 완성시켜서 R|T를 생성
            -BA를 통해 다른 frame과의 RER 비교를 통해 global 데이터를 보정함
            -noise가 많이 섞인 경우는 pnp solver를 통한 RT의 신뢰도가 낮음
            -설계값을 모르는 경우에도 방안 생간
        '''
        gathering_data_single(ax1, script_dir, bboxes, START_FRAME, STOP_FRAME, 0, 0)
        BA_RT(info_name='CAMERA_INFO.pickle', save_to='BA_RT.pickle', target='BLENDER') 
        
        # 2차 보정
        gathering_data_single(ax1, script_dir, bboxes, START_FRAME, STOP_FRAME, 0, 1)
        remake_3d_for_blob_info(blob_cnt=BLOB_CNT, info_name='BLOB_INFO.pickle', undistort=undistort, opencv=DONE, blender=DONE, ba_rt=DONE)
        # BA_3D_POINT(RT='BLENDER')  // 사용 안함

        '''
        ToDO
        부분 LSM 안됨
        '''
        LSM(TARGET_DEVICE, MODEL_DATA)
        gathering_data_single(ax1, script_dir, bboxes, START_FRAME, STOP_FRAME, 1, 1)
    
        draw_result(MODEL_DATA, ax1=ax1, ax2=ax2, opencv=DONE, blender=DONE, ba_rt=DONE)
        Check_Calibration_data_combination(combination_cnt)
        
        '''
        SEED PATH 저장
        ''' 
        # save_camera_position(TARGET_DEVICE)

    elif SOLUTION == 2:
        gathering_data_single(ax1, script_dir, bboxes, START_FRAME, STOP_FRAME, 0, 0)
        remake_3d_for_blob_info(blob_cnt=BLOB_CNT, info_name='BLOB_INFO.pickle', undistort=undistort, opencv=DONE, blender=DONE, ba_rt=NOT_SET)
        # BA_3D_POINT(RT='BLENDER')
        draw_result(MODEL_DATA, ax1=ax1, ax2=ax2, opencv=DONE, blender=DONE, ba_rt=NOT_SET, ba_3d=NOT_SET)
    else:
        print('Do Nothing')
        

    if SHOW_PLOT == 1:
        plt.show()

    print('\n\n')
    print('########## DONE ##########')