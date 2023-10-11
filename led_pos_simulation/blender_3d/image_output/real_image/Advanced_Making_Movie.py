from Advanced_Function import *

TRACKER_PADDING = 2
# BLOB_SIZE = 150
BLOB_SIZE = 1000

def make_video(blob_file, camera_devices):
    START_PRINT_RAWDATA = NOT_SET
    BLOB_SETTING = NOT_SET
    TEMP_POS = {'status': NOT_SET, 'mode': RECTANGLE, 'start': [], 'move': [], 'circle': NOT_SET, 'rectangle': NOT_SET}
    POS = {}
    bboxes = []    
    json_file = os.path.join(script_dir, blob_file)
    json_data = rw_json_data(READ, json_file, None)
    if json_data != ERROR:
        bboxes = json_data['bboxes']
        if 'mode' in json_data:
            TEMP_POS['mode'] = json_data['mode']
            if TEMP_POS['mode'] == RECTANGLE:
                TEMP_POS['rectangle'] = json_data['roi']
            else:
                TEMP_POS['circle'] = json_data['roi']        
            POS = TEMP_POS

    # Select the first camera device
    camera_port = camera_devices[0]['port']

    # Open the video capture
    cap = cv2.VideoCapture(camera_port)

    # Set the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    # fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    # out = cv2.VideoWriter('output.mkv', fourcc, 60.0, (1280, 960))

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' instead of 'X264'
    # out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1280, 960))
    recording = False


    # Initialize jitter dictionary and lines dictionary
    jitters = {}
    prev_blobs = {}
    detected = 0
    if cap.isOpened():
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            draw_frame = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, frame = cv2.threshold(frame, 200, 255, cv2.THRESH_TOZERO)
            if not ret:
                print("Unable to capture video")
                break

            # Convert the frame to grayscale
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # center_x, center_y = 1280 // 2, 960 // 2
            # cv2.line(draw_frame, (0, center_y), (1280, center_y), (255, 255, 255), 1)
            # cv2.line(draw_frame, (center_x, 0), (center_x, 960), (255, 255, 255), 1)     

            # Start/Stop recording
            filtered_blob_area = []
            if BLOB_SETTING == DONE:
                blob_area = detect_led_lights(frame, TRACKER_PADDING)
                for _, bbox in enumerate(blob_area):
                    (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                    gcx,gcy, gsize = find_center(frame, (x, y, w, h))
                    if gsize < BLOB_SIZE:
                        continue
                    filtered_blob_area.append((gcx, gcy, (x, y, w, h)))        

                cv2.namedWindow('image')
                partial_click_event = functools.partial(click_event, frame=frame, blob_area_0=filtered_blob_area, bboxes=bboxes, POS=TEMP_POS)
                cv2.setMouseCallback('image', partial_click_event)        

            if START_PRINT_RAWDATA == DONE:
                detected += 1
                for box in bboxes:
                    (x, y, w, h) = box['bbox']
                    id = int(box['idx'])
                    # print(f"{id} rawdata")
                    gcx,gcy, _ = find_center(frame, (x, y, w, h))
                    if id in prev_blobs:
                        prev_x, prev_y = prev_blobs[id]
                        jitter = ((gcx - prev_x)**2 + (gcy - prev_y)**2)**0.5
                        if id not in jitters:
                            jitters[id] = []
                        jitters[id].append(jitter)
                    prev_blobs[id] = (gcx, gcy)

                    if 0:
                        cropped = crop_image(frame, int(x), int(y), int(w), int(h))                    
                        # Cropped 이미지의 데이터만 깔끔하게 프린트                    
                        for row in cropped:
                            print(' '.join(f"{pixel:5}" for pixel in row))
                
                cv2.putText(draw_frame, f"detected {detected}", (CAP_PROP_FRAME_WIDTH - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


            if TEMP_POS['status'] == MOVE:
                dx = np.abs(TEMP_POS['start'][0] - TEMP_POS['move'][0])
                dy = np.abs(TEMP_POS['start'][1] - TEMP_POS['move'][1])
                if TEMP_POS['mode'] == CIRCLE:
                    radius = math.sqrt(dx ** 2 + dy ** 2) / 2
                    cx = int((TEMP_POS['start'][0] + TEMP_POS['move'][0]) / 2)
                    cy = int((TEMP_POS['start'][1] + TEMP_POS['move'][1]) / 2)
                    # print(f"{dx} {dy} radius {radius}")
                    TEMP_POS['circle'] = [cx, cy, radius]
                else:                
                    TEMP_POS['rectangle'] = [TEMP_POS['start'][0],TEMP_POS['start'][1],TEMP_POS['move'][0],TEMP_POS['move'][1]]
        
            elif TEMP_POS['status'] == UP:
                print(TEMP_POS)
                TEMP_POS['status'] = NOT_SET        
            if TEMP_POS['circle'] != NOT_SET and TEMP_POS['mode'] == CIRCLE:
                cv2.circle(draw_frame, (TEMP_POS['circle'][0], TEMP_POS['circle'][1]), int(TEMP_POS['circle'][2]), (255,255,255), 1)
            elif TEMP_POS['rectangle'] != NOT_SET and TEMP_POS['mode'] == RECTANGLE:
                cv2.rectangle(draw_frame,(TEMP_POS['rectangle'][0],TEMP_POS['rectangle'][1]),(TEMP_POS['rectangle'][2],TEMP_POS['rectangle'][3]),(255,255,255),1)

            key = cv2.waitKey(1)

            if key == ord('c'):
                print('clear area')
                bboxes.clear()
                if TEMP_POS['mode'] == RECTANGLE:
                    TEMP_POS['rectangle'] = NOT_SET
                else:
                    TEMP_POS['circle'] = NOT_SET
            elif key == ord('s'):
                print('save blob area')
                json_data = OrderedDict()
                json_data['bboxes'] = bboxes
                json_data['mode'] = TEMP_POS['mode']
                json_data['roi'] = TEMP_POS['circle'] if TEMP_POS['mode'] == CIRCLE else TEMP_POS['rectangle']
                POS = TEMP_POS
                # Write json data
                rw_json_data(WRITE, json_file, json_data)
            elif key & 0xFF == 27:
                print('ESC pressed')
                cv2.destroyAllWindows()
                sys.exit()
            elif key == ord('q'):
                break
            elif key == ord('p'):
                START_PRINT_RAWDATA = DONE
            elif key == ord('m'):            
                if TEMP_POS['mode'] == RECTANGLE:
                    TEMP_POS['mode'] = CIRCLE
                else:
                    TEMP_POS['mode'] = RECTANGLE
                print('MODE changed ', TEMP_POS['mode'])
                

            # if key & 0xFF == ord('s'):
            #     if not recording:
            #         print('Start Recording')                
            #         recording = True
            # elif key & 0xFF == ord('q'):
            #     if recording:
            #         print('Stop Recording')
            #         out.release()
            #         recording = False
            # elif key & 0xFF == ord('e'): 
            #     # Use 'e' key to exit the loop
            #     break
            # elif key & 0xFF == ord('c'): 
            #     # Capture current frame
            #     cv2.imwrite('start_capture.jpg', frame)
            #     print('Image saved as capture.jpg')
            # # Write the frame to file if recording
            # if recording and out is not None:
            #     print('writing...')
            #     out.write(frame)
            

            # Display the resulting frame
            draw_blobs_and_ids(draw_frame, filtered_blob_area, bboxes)
            cv2.imshow('image', draw_frame)

    # if out is not None:
    #     out.release()
    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()

    return jitters

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(os.getcwd())


    cam_dev_list = terminal_cmd('v4l2-ctl', '--list-devices')
    camera_devices = init_model_json(cam_dev_list)
    print(camera_devices)
    
    jitters = make_video(f"{script_dir}/jsons/test_3/make_movie.json", camera_devices)

    # 각 ID에 대한 jitter 값의 표준 편차와 평균을 계산
    for id, jitter_values in jitters.items():
        std_value = np.std(jitter_values) * 6
        mean_value = np.mean(jitter_values)
        print(f'ID: {id}, STD: {std_value:.8f}, Mean: {mean_value:.8f}')