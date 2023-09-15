from Advanced_Function import *
from Advanced_Calibration import blob_setting
from Advanced_Correspondence_Class import *

# Common Functions
def find_center(frame, SPEC_AREA):
    x_sum = 0
    t_sum = 0
    y_sum = 0
    g_c_x = 0
    g_c_y = 0
    m_count = 0

    (X, Y, W, H) = SPEC_AREA

    for y in range(Y, Y + H):
        for x in range(X, X + W):
            if y < 0 or y >= CAP_PROP_FRAME_HEIGHT or x < 0 or x >= CAP_PROP_FRAME_WIDTH:
                continue
            x_sum += x * frame[y][x]
            t_sum += frame[y][x]
            if frame[y][x] > 0:
                m_count += 1

    for x in range(X, X + W):
        for y in range(Y, Y + H):
            if y < 0 or y >= CAP_PROP_FRAME_HEIGHT or x < 0 or x >= CAP_PROP_FRAME_WIDTH:
                continue
            y_sum += y * frame[y][x]

    if t_sum != 0:
        g_c_x = x_sum / t_sum
        g_c_y = y_sum / t_sum

    if g_c_x == 0 or g_c_y == 0:
        return 0, 0, 0

    return round(g_c_x, 8), round(g_c_y, 8), m_count
def detect_led_lights(image, padding=5):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blob_info = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # Apply padding for the bounding box
        x -= padding
        y -= padding
        w += padding * 2
        h += padding * 2

        # Use contourArea to get the actual area of the contour
        # area = cv2.contourArea(contour)

        # Check if the area of the contour is within the specified range
        blob_info.append((x, y, w, h))

    return blob_info
def blob_area_setting(blob_file, image_files):
    print('blob_setting START')
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

    frame_cnt = 0
    while True:
        if frame_cnt >= len(image_files):
            break
        frame = cv2.imread(image_files[frame_cnt])
        filename = f"IMAGE Mode {os.path.basename(image_files[frame_cnt])}"
        if frame is None:
            print("Cannot read the first image")
            cv2.destroyAllWindows()
            exit()

        draw_frame = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape
        center_x, center_y = width // 2, height // 2
        cv2.line(draw_frame, (0, center_y), (width, center_y), (255, 0, 0), 1)
        cv2.line(draw_frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)       
        cv2.putText(draw_frame, f"frame_cnt {frame_cnt} [{filename}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)

        cv2.namedWindow('image')
        partial_click_event = functools.partial(click_event, frame=frame, blob_area_0=NOT_SET, bboxes=bboxes, POS=TEMP_POS)
        cv2.setMouseCallback('image', partial_click_event)
        
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
        elif key == ord('n'):
            frame_cnt += 1
            bboxes.clear()
        elif key == ord('b'):
            frame_cnt -= 1
            bboxes.clear()
        elif key == ord('m'):            
            if TEMP_POS['mode'] == RECTANGLE:
                TEMP_POS['mode'] = CIRCLE
            else:
                TEMP_POS['mode'] = RECTANGLE
            print('MODE changed ', TEMP_POS['mode'])


        cv2.imshow('image', draw_frame)

    print('done')
    cv2.destroyAllWindows()

    return bboxes, POS
def area_filter(areas, img):
    # 영역 추출
    x1, y1, x2, y2 = areas['rectangle']
    # 빈 마스크 생성
    mask = np.zeros_like (img)
    # 마스크에 사각형 영역을 흰색으로 채움
    mask[y1:y2, x1:x2] = 255
    # 원본 이미지와 마스크 합성
    result = cv2.bitwise_and (img, mask)

    return result
def draw_result_chart(results):
    fig, axes = plt.subplots (1, 2, figsize=(14, 6))
    for idx, jitters in enumerate([results[0], results[1]]):
        ax = axes[idx]
        ids = []
        std_values = []
        mean_values = []
        for id, jitter_values in jitters.items ():
            std_value = np.std (jitter_values) * 6
            mean_value = np.mean (jitter_values)
            print(f"ID {id} - Std(6): {std_value}, Mean: {mean_value}")

            ids.append (str(id))
            std_values.append (std_value)
            mean_values.append (mean_value)

        # 막대 그래프 생성
        x = np.arange (len(ids))
        width = 0.35

        rects1 = ax.bar (x - width/2, std_values, width, label='Std(6)')
        rects2 = ax.bar (x + width/2, mean_values, width, label='Mean')

        # 레이블, 타이틀, 범례 추가
        ax.set_ylabel ('Values')
        ax.set_xlabel ('IDs')
        ax.set_title (f'Std and Mean values by ID for dataset {idx + 1}')
        ax.set_xticks (x)
        ax.set_xticklabels (ids)
        ax.legend ()

    plt.tight_layout ()
    plt.show ()

####################################### Filter TEST #######################################
def GaussianBlur(img):
    '''
        1. threshod cut
        2. gaussian blur
    '''
    CV_MAX_THRESHOLD = 255
    CV_MIN_THRESHOLD = 10
    GAUSSIAN_KERNEL = (3, 3)
    GAUSSIAN_SIG = 1.0

    _, img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
    img_tmp = cv2.GaussianBlur(img, GAUSSIAN_KERNEL, GAUSSIAN_SIG)

    DEBUG_LOG = f"W:{CAP_PROP_FRAME_WIDTH} H:{CAP_PROP_FRAME_HEIGHT}" + '\n' \
                f"Gaussian F:{GAUSSIAN_KERNEL} {GAUSSIAN_SIG}" + '\n' + \
                f"Thres:{CV_MIN_THRESHOLD}"

    return img_tmp, DEBUG_LOG


def GaussianSharp(img):
    '''
        1. threshod cut
        2. gaussian blur
        3. sharpness filter
        4. addweight
    '''
    CV_MAX_THRESHOLD = 255
    CV_MIN_THRESHOLD = 20
    GAUSSIAN_KERNEL = (3, 3)
    GAUSSIAN_SIG = 1.0
    SHARPNESS_KERNEL = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
    ADD_WEIGHT = (1.0, -0.1)
    
    _, img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
    blurred = cv2.GaussianBlur(img, GAUSSIAN_KERNEL, GAUSSIAN_SIG)
    sharpened = cv2.filter2D(blurred, -1, SHARPNESS_KERNEL)
    blur_sharp_img = cv2.addWeighted(blurred, ADD_WEIGHT[0], sharpened,ADD_WEIGHT[1], 0)

    DEBUG_LOG = f"W:{CAP_PROP_FRAME_WIDTH} H:{CAP_PROP_FRAME_HEIGHT}" + '\n' \
                f"Gaussian F:{GAUSSIAN_KERNEL} {GAUSSIAN_SIG}" + '\n' + \
                f"SHARPNESS K:\n{SHARPNESS_KERNEL}" + '\n' + \
                f"ADD_WEIGHT :{ADD_WEIGHT}" + '\n' + \
                f"Thres:{CV_MIN_THRESHOLD}"

    stacked_frame = np.hstack ((blurred, sharpened, blur_sharp_img))
    cv2.imshow('GaussianSharp', stacked_frame)

    return blur_sharp_img, DEBUG_LOG
####################################### Filter TEST #######################################


####################################### Segmentation TEST #######################################
# Algorithm 1
def segmentation_algo_1(image, blob_area, max_level):
    CANNY_THRESHOD = 100
    # print(blob_area)
    (x, y, w, h) = blob_area

    ratio = w / h
    if 0.8 <= ratio <= 1.2:
        return NOT_SET, 0

    cropped = crop_image(image, x, y, w, h)
    # print(f"shape: {w} {h}")
    # Generate image pyramid using pyrUp (increasing resolution)
    gaussian_pyramid = [cropped]
    for i in range(max_level):
        cropped = cv2.pyrUp(cropped)
        ch, cw = cropped.shape[:2]
        if min(ch, cw) > CAP_PROP_FRAME_HEIGHT:
            break
        # print(f"pryUP {ch} {cw}")
        gaussian_pyramid.append(cropped)

    image = gaussian_pyramid[len(gaussian_pyramid) - 1]
    # blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    ret, threshold_image = cv2.threshold(image, CANNY_THRESHOD, 255, cv2.THRESH_BINARY)

    # cv2.imshow('threshold_image',threshold_image)
    # cv2.waitKey(0)
    # contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # c0 = contours[0]
    # AREA = cv2.contourArea(c0)
    # print(f"contour area: {AREA}")

    # 경계선 감지
    edges = cv2.Canny(threshold_image, CANNY_THRESHOD, 255)

    # 팽창과 침식을 조합하여 경계선 부드럽게 만들기
    kernel = np.ones((10, 10), np.uint8)
    smoothed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('smoothed_edges',smoothed_edges)
    # cv2.waitKey(0)
    # Copy the thresholded image.
    im_floodfill = smoothed_edges.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = smoothed_edges.shape[:2]
    # print(f"smoothed_edges: {h} {w}")
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground.
    im_out = smoothed_edges | im_floodfill_inv

    # laplacian = cv2.Laplacian(threshold_image,cv2.CV_8U,ksize=5)
    image = cv2.cvtColor(im_out, cv2.COLOR_GRAY2BGR)
    draw_frame = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    th, bw = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    borderSize = int(max(h, w) / 8)

    distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize, 
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    gap = 2

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(borderSize-gap)+1, 2*(borderSize-gap)+1))
    kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap, 
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
    mn, mx, _, _ = cv2.minMaxLoc(nxcor)
    th, peaks = cv2.threshold(nxcor, mx*0.5, 255, cv2.THRESH_BINARY)
    peaks8u = cv2.convertScaleAbs(peaks)
    contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    peaks8u = cv2.convertScaleAbs(peaks)

    # print(f"len(contours) {len(contours)}")
    R_DETECT = 0
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        _, mx, _, mxloc = cv2.minMaxLoc(dist[y:y+h, x:x+w], peaks8u[y:y+h, x:x+w])
        if int(mx) <= 0:
            continue
        cv2.circle(draw_frame, (int(mxloc[0]+x), int(mxloc[1]+y)), int(mx), (255, 0, 0), 1)
        # print(f"center:{(int(mxloc[0]+x), int(mxloc[1]+y))} R:{int(mx)}")
        # cv2.rectangle(draw_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.drawContours(draw_frame, contours, i, (0, 0, 255), 1)
        R_DETECT += 1

    return draw_frame, R_DETECT

####################################### Segmentation TEST #######################################

def blob_detection(img):
    # Segmentation Test
    MIN_BLOB_SIZE = 2
    MAX_BLOB_SIZE = 100
    TRACKER_PADDING = 1
    CV_MAX_THRESHOLD = 255
    CV_MIN_THRESHOLD = 95
    _, img = cv2.threshold(img, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)

    blob_area = detect_led_lights(img, TRACKER_PADDING)
    blobs = []
    for blob_id, bbox in enumerate(blob_area):
        (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        gcx, gcy, gsize = find_center(img, (x, y, w, h))
        if gsize < MIN_BLOB_SIZE or gsize > MAX_BLOB_SIZE:
            continue            
        
        blobs.append([-1, gcx, gcy, gsize, w, h])

        if DO_SEG == 1:
            result_frame, R_DETECT = segmentation_algo_1(img, (x, y, w, h), max_level=3)
            if R_DETECT > 1:
                cv2.putText(img, f"SEG", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.imshow('SEG frame', result_frame)
                continue
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
            
    blobs = sorted(blobs, key=lambda x:(x[1], x[2])) ## 또는 l.sort(key=lambda x:x[1])

    for idx, blob in enumerate(blobs):
        # print(blob)
        blob[0] = idx
        (x, y) = (int(blob[1]), int(blob[2]))
        cv2.putText(img, f"{blob[0]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    ret_blobs = copy.deepcopy(blobs)
    return ret_blobs, img
####################################### Segmentation TEST #######################################



def read_image(IMAGE_FILES):
    image_files = IMAGE_FILES[0]
    areas = IMAGE_FILES[1]
    print('lenght of images: ', len(image_files))
    print(f"areas: {areas}")

    # Initialize jitter dictionary and lines dictionary
    jitters = {}
    lines = {}
    prev_blobs = {}
    # Initialize the number of subplots based on initial ID count

    initial_id_count = 10 # 예시 값
    first_detection = 0

    if SHOW_CHART == 1:
        fig, axs = plt.subplots (initial_id_count, 1, sharex=True)
        for idx, ax in enumerate(axs):
            ax.set_ylim([0, 0.5])
            ax.set_title(f'ID{idx}')
        fig.set_size_inches(10, 15)
        fig.suptitle("Jitter")
        plt.ion ()

    curr_frame_cnt = 0
    prev_frame_cnt = -1


    while True:
        if curr_frame_cnt >= len(image_files):
            break
        frame_0 = cv2.imread(image_files[curr_frame_cnt])
        filename = f"IMAGE Mode {os.path.basename(image_files[curr_frame_cnt])}"
        if frame_0 is None or frame_0.size == 0:
            print(f"Failed to load {image_files[curr_frame_cnt]}, frame_cnt:{curr_frame_cnt}")
            continue
        draw_frame = frame_0.copy()

        # Area Bypass Filter
        bypass_img = area_filter(areas, frame_0)
        # Make Filter IMAGE
        TEST_IMAGE, DEBUG_LOG = GaussianSharp(bypass_img)

        # BLOB Detection
        blobs, TEST_IMAGE = blob_detection(TEST_IMAGE)

        # ID MAPPING
        if DO_LONGSEARCH == 1 and first_detection == 0:
            if len(blobs) > 4:
                points2D_D = []
                for blob in blobs:
                    points2D_D.append(blob[1:3])
                points2D_D = np.array(points2D_D, dtype=np.float64)
                print(f"points2D_D\n {points2D_D }")
                SEARCH_BLOBS, points2D_U = correspondence_search_set_blobs(points2D_D, pebble_camera_matrix, blobs=blobs)
                DATA_SET = (MODEL_DATA, DIRECTION, SEARCH_MODEL, SEARCH_BLOBS, pebble_camera_matrix, points2D_D, points2D_U)
                long_search_python(DATA_SET)
                first_detection = 1
        
        
        if curr_frame_cnt > prev_frame_cnt and len(blobs) > 0:
            for blob in blobs:
                id, x, y, _, _, _ = blob
                if id in prev_blobs:
                    prev_x, prev_y = prev_blobs[id]
                    jitter = ((x - prev_x)**2 + (y - prev_y)**2)**0.5
                    if id not in jitters:
                        jitters[id] = []
                    jitters[id].append(jitter)

                    # Update the line data for this ID
                    if SHOW_CHART == 1:
                        if id < initial_id_count:
                            if id not in lines:
                                lines[id], = axs[id].plot(jitters[id], label=f'ID {id}')
                            else:
                                lines[id].set_ydata(jitters[id])
                                lines[id].set_xdata(range(len(jitters[id])))
                        axs[id].relim()
                        axs[id].autoscale_view()
                prev_blobs[id] = (x, y)

            if SHOW_CHART:
                plt.draw()
                plt.pause(0.01)


        prev_frame_cnt = curr_frame_cnt

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            if SHOW_CHART == 1:
                plt.close()
            break
        elif key & 0xFF == ord('n'):
                if AUTO_LOOP == 0:
                    curr_frame_cnt += 1     
        elif key & 0xFF == ord('b'):
                if AUTO_LOOP == 0:
                    curr_frame_cnt -= 1    

        if AUTO_LOOP:
            curr_frame_cnt += 1


        # DEBUG and Show IMAGE
        cv2.putText(draw_frame, f"frame_cnt:{curr_frame_cnt} file:[{filename}]", (10, 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(draw_frame, f"ORIGIN", (CAP_PROP_FRAME_WIDTH - 100, 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)   

        # Display the resulting frame        
        TEST_IMAGE = cv2.cvtColor(TEST_IMAGE, cv2.COLOR_GRAY2BGR)
        debug_lines = DEBUG_LOG.split ('\n')
        y0, dy = 10, 15 # 시작 y좌표와 라인 간의 거리

        for i, line in enumerate(debug_lines):
            y = y0 + i * dy
            cv2.putText (TEST_IMAGE, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX , 0.4, (0, 255, 0), 1)
        cv2.putText(TEST_IMAGE, f"FILTERED", (CAP_PROP_FRAME_WIDTH - 100, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(TEST_IMAGE, f"Detected {len(blobs)}", (CAP_PROP_FRAME_WIDTH - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.rectangle(TEST_IMAGE,(areas['rectangle'][0], areas['rectangle'][1]),(areas['rectangle'][2],areas['rectangle'][3]),(255,255,255),1)


        # show Image
        stacked_frame = np.hstack ((draw_frame, TEST_IMAGE))
        cv2.imshow('IMAGE', stacked_frame)

    cv2.destroyAllWindows()

    return jitters



if __name__ == "__main__":
    pebble_camera_matrix = [
        # 0번 sensor
        [np.array([[240.699213, 0.0, 313.735554],
                [0.0, 240.394949, 235.316344],
                [0.0, 0.0, 1.0]], dtype=np.float64),
        np.array([[0.040384], [-0.015174], [-0.000401], [-0.000584]], dtype=np.float64)],
    ]

    SHOW_CHART = 0
    AUTO_LOOP = 0
    DO_SEG = 1
    DO_LONGSEARCH = 1
    CAP_PROP_FRAME_WIDTH = 640
    CAP_PROP_FRAME_HEIGHT = 480

    MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/arcturas_new/arcturas_left_#4_new.json"))
    MODEL_DATA = np.array(MODEL_DATA, dtype=np.float64)
    DIRECTION = np.array(DIRECTION, dtype=np.float64)

    BLOB_CNT = len(MODEL_DATA)
    print('PTS')
    for i, leds in enumerate(MODEL_DATA):
        print(f"{np.array2string(leds, separator=', ')},")
    print('DIR')
    for i, dir in enumerate(DIRECTION):
        print(f"{np.array2string(dir, separator=', ')},")

    _, SEARCH_MODEL = led_search_candidate_new(MODEL_DATA, DIRECTION)

    RESULTS = []
    for i in range(1):
        # image_files = sorted(glob.glob(f"{script_dir}/../../../../../dataset/dataset_segmentation/front_side/dataset_front_70cm/CAM{i}*.bmp"))
        image_files = sorted(glob.glob(f"{script_dir}/../../../../../dataset/dataset_segmentation/left_side/dataset_left_70cm/CAM{i}*.bmp"))

        # image_files = sorted(glob.glob(f"{script_dir}/../../../../../dataset/dataset_segmentation/left_side/dataset_left_40cm/CAM{i}*.bmp"))
    
        _, areas = blob_area_setting(f"{script_dir}/jsons/test_3/blob_area_{i}.json", image_files)
        jitters = read_image((image_files, areas))
        RESULTS.append(jitters)

    # draw_result_chart(RESULTS)

