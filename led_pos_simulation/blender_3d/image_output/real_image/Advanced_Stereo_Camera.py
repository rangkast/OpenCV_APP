from Advanced_Function import *

MODEL_DATA, DIRECTION = init_coord_json(f"{script_dir}/jsons/specs/semi_slam_polyhedron.json")
BLOB_CNT = len(MODEL_DATA)
CV_MAX_THRESHOLD = 255
CV_MIN_THRESHOLD = 100
BLOB_SIZE = 50
TRACKER_PADDING = 10


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

def stereo_blob_setting(camera_devices):
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
        ret1, frame1 = cam_L.read()
        ret2, frame2 = cam_R.read()
        if not ret1 or not ret2:
            break

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
            rw_json_data(WRITE, f"{script_dir}/render_img/stereo/blob_area_left.json", json_data)

            json_data = OrderedDict()
            json_data['bboxes'] = right_bboxes
            rw_json_data(WRITE, f"{script_dir}/render_img/stereo/blob_area_right.json", json_data)

    cam_L.release()
    cam_R.release()
    cv2.destroyAllWindows()

    return left_bboxes, right_bboxes


# def calibration(left_bboxes, right_bboxes):



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
    
    show_calibrate_data(np.array(MODEL_DATA), np.array(DIRECTION))

    left_bboxes, right_bboxes = stereo_blob_setting(camera_devices)

    
    



