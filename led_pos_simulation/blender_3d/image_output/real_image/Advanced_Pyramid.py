from Advanced_Function import *

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

MODEL_DATA, DIRECTION = init_coord_json(f"{script_dir}/jsons/specs/rifts_left_2.json")
TARGET_DEVICE = 'RIFTS'
VIDEO_MODE = 1
CV_MAX_THRESHOLD = 255
CV_MIN_THRESHOLD = 100
BLOB_SIZE = 10
TRACKER_PADDING = 10
max_level = 5
AUTO_LOOP = 0


def find_circles_or_ellipses(image, draw_frame):
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    # Apply threshold
    ret, threshold_image = cv2.threshold(blurred_image, 150, 255, cv2.THRESH_BINARY)
    # Perform Edge detection on the thresholded image
    edges = cv2.Canny(threshold_image, 150, 255)
    padding = 2
    height, width = image.shape[:2]
    max_radius = int(min(width, height) / 2 - padding)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=50, minRadius=0,
                               maxRadius=max_radius)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # c0 = contours[0]
    # AREA = cv2.contourArea(c0)

    # print(f"contour area: {AREA}")


    circle_count = 0
    ellipse_count = 0

    if circles is not None:
        for circle in circles[0, :]:
            # Convert the center coordinates to int
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])
            cv2.circle(draw_frame, center, radius, (255, 0, 0), 1)
            circle_count += 1

    for contour in contours:
        if len(contour) >= 5:  # A contour must have at least 5 points for fitEllipse
            ellipse = cv2.fitEllipse(contour)
            if ellipse[1][0] > 0 and ellipse[1][1] > 0:  # width and height must be positive
                cv2.ellipse(draw_frame, ellipse, (0, 0, 255), 1)
                ellipse_count += 1

    # Draw all contours for debugging purposes
    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

    # Check if we found multiple blobs or not a circle/ellipse
    if circle_count + ellipse_count > 1:
        print(f"Detected {circle_count} circles and {ellipse_count} ellipses")
        return True, image, draw_frame

    return False, image, draw_frame
def check_blobs_with_pyramid(image, draw_frame, x, y, w, h, max_level):
    # Crop the image
    cropped = crop_image(image, x, y, w, h)

    # Generate image pyramid using pyrUp (increasing resolution)
    gaussian_pyramid = [cropped]
    draw_pyramid = [crop_image(draw_frame, x, y, w, h)]  # Also generate pyramid for draw_frame
    for i in range(max_level):
        cropped = cv2.pyrUp(cropped)
        draw_frame_cropped = cv2.pyrUp(draw_pyramid[-1])
        gaussian_pyramid.append(cropped)
        draw_pyramid.append(draw_frame_cropped)

    FOUND_STATUS = False
    # Check for circles or ellipses at each level
    for i, (img, draw_frame_cropped) in enumerate(zip(gaussian_pyramid, draw_pyramid)):
        found, img_with_contours, draw_frame_with_shapes = find_circles_or_ellipses(img.copy(), draw_frame_cropped.copy())
        # Save the image for debugging
        if found:
            FOUND_STATUS = True
            save_image(img_with_contours, f"debug_{i}_{FOUND_STATUS}")
            save_image(draw_frame_with_shapes, f"debug_draw_{i}_{FOUND_STATUS}")
        
    return FOUND_STATUS
def advanced_check_contours(image, draw_frame, shape):
    edges = cv2.Canny(image, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c0 = contours[0]
    AREA = cv2.contourArea(c0)
    print(f"contour area: {AREA}")

    for contour in contours:
        if len(contour) >= 5:  # A contour must have at least 5 points for fitEllipse
            ellipse = cv2.fitEllipse(contour)
            ellipse = ((ellipse[0][0], ellipse[0][1]), ellipse[1], ellipse[2])  # 타원의 중심 위치를 보정합니다.
            center, (major_axis, minor_axis), angle = ellipse
            if major_axis > 0 and minor_axis > 0:
                cv2.ellipse(draw_frame, ellipse, (0, 0, 255), 1)
                print(f"major_axis:{major_axis} minor_axis:{minor_axis}")

    # Draw all contours for debugging purposes
    cv2.drawContours(draw_frame, contours, -1, (0, 255, 0), 1)

    '''
    Parameters:	
    image 8-bit single-channel image. grayscale image.
    method 검출 방법. 현재는 HOUGH_GRADIENT가 있음.
    dp dp=1이면 Input Image와 동일한 해상도.
    minDist 검출한 원의 중심과의 최소거리. 값이 작으면 원이 아닌 것들도 검출이 되고, 너무 크면 원을 놓칠 수 있음.
    param1  내부적으로 사용하는 canny edge 검출기에 전달되는 Paramter
    param2  이 값이 작을 수록 오류가 높아짐. 크면 검출률이 낮아짐.
    minRadius 원의 최소 반지름.
    maxRadius 원의 최대 반지름.
    '''
    padding = 1
    max_radius = int(min(shape[0], shape[1]) / 2) - padding
    min_radius = int(max_radius / 2) - padding
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1, param1=50, param2=20,
                            minRadius=min_radius,
                            maxRadius=max_radius)
    if circles is not None:
        for circle in circles[0, :]:
            # Convert the center coordinates to int
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])
            cv2.circle(draw_frame, center, radius, (255, 0, 0), 1)
def test_code_1(image, shape):
    print(shape)
    (x, y, w, h) = shape
    cropped = crop_image(image, x, y, w, h)

    # Generate image pyramid using pyrUp (increasing resolution)
    gaussian_pyramid = [cropped]
    for i in range(max_level):
        cropped = cv2.pyrUp(cropped)
        ch, cw = cropped.shape[:2]
        print(f"{ch} {cw}")
        if min(ch, cw) > 480:
            break
        gaussian_pyramid.append(cropped)

    image = gaussian_pyramid[len(gaussian_pyramid) - 1]
    # blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    ret, threshold_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

    cv2.imshow('threshold_image',threshold_image)
    cv2.waitKey(0)
    # 경계선 감지
    edges = cv2.Canny(threshold_image, 100, 255)

    # 팽창과 침식을 조합하여 경계선 부드럽게 만들기
    kernel = np.ones((10, 10), np.uint8)
    smoothed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('smoothed_edges',smoothed_edges)
    cv2.waitKey(0)
    # Copy the thresholded image.
    im_floodfill = smoothed_edges.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = smoothed_edges.shape[:2]
    print(f"h,w: {h} {w}")
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

    borderSize = 30
    distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize, 
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    gap = 5
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(borderSize-gap)+1, 2*(borderSize-gap)+1))
    kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap, 
                                    cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
    mn, mx, _, _ = cv2.minMaxLoc(nxcor)
    th, peaks = cv2.threshold(nxcor, mx*0.5, 255, cv2.THRESH_BINARY)
    peaks8u = cv2.convertScaleAbs(peaks)
    contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    peaks8u = cv2.convertScaleAbs(peaks)    # to use as mask
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        _, mx, _, mxloc = cv2.minMaxLoc(dist[y:y+h, x:x+w], peaks8u[y:y+h, x:x+w])
        cv2.circle(draw_frame, (int(mxloc[0]+x), int(mxloc[1]+y)), int(mx), (255, 0, 0), 2)
        cv2.rectangle(draw_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.drawContours(draw_frame, contours, i, (0, 0, 255), 2)

    return draw_frame

def test_code_2(image, draw_frame):
    from scipy.ndimage import label

    pi_4 = 4*math.pi

    def segment_on_dt(img):
        border = img - cv2.erode(img, None)

        dt = cv2.distanceTransform(255 - img, 2, 3)
        dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
        _, dt = cv2.threshold(dt, 100, 255, cv2.THRESH_BINARY)

        lbl, ncc = label(dt)
        lbl[border == 255] = ncc + 1

        lbl = lbl.astype(np.int32)
        cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), lbl)
        lbl[lbl < 1] = 0
        lbl[lbl > ncc] = 0

        lbl = lbl.astype(np.uint8)
        lbl = cv2.erode(lbl, None)
        lbl[lbl != 0] = 255
        return lbl

    def find_circles(frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 2)

        edges = frame_gray - cv2.erode(frame_gray, None)
        _, bin_edge = cv2.threshold(edges, 0, 255, cv2.THRESH_OTSU)
        height, width = bin_edge.shape
        mask = np.zeros((height+2, width+2), dtype=np.uint8)
        cv2.floodFill(bin_edge, mask, (0, 0), 255)

        components = segment_on_dt(bin_edge)

        circles, obj_center = [], []
        contours, _ = cv2.findContours(components,
                cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            c = c.astype(np.int64) # XXX OpenCV bug.
            area = cv2.contourArea(c)
            if 100 < area < 3000:
                arclen = cv2.arcLength(c, True)
                circularity = (pi_4 * area) / (arclen * arclen)
                if circularity > 0.5: # XXX Yes, pretty low threshold.
                    circles.append(c)
                    box = cv2.boundingRect(c)
                    obj_center.append((box[0] + (box[2] / 2), box[1] + (box[3] / 2)))

        return circles, obj_center

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    circles, new_center = find_circles(image)
    for i in range(len(circles)):
        cv2.drawContours(draw_frame, circles, i, (0, 255, 0))
def get_blob_area(frame):
    filtered_blob_area = []
    blob_area = detect_led_lights(frame, TRACKER_PADDING)
    for _, bbox in enumerate(blob_area):
        (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        gcx,gcy, gsize = find_center(frame, (x, y, w, h))        
        if gsize < BLOB_SIZE:
            continue        
        filtered_blob_area.append((gcx, gcy, (x, y, w, h)))   

    return filtered_blob_area
def pyramid_test(camera_devices):
    if VIDEO_MODE == 1:
        cam_L = cv2.VideoCapture(camera_devices[0]['port'])
        cam_L.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
        cam_L.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)

    else:
        image_files = sorted(glob.glob(f"{script_dir}/pyramid_test/*.png"))
        print(image_files)

    frame_cnt = 0
    DO_PYRAMID = 0
    while True:
        if VIDEO_MODE == 1:
            ret1, frame1 = cam_L.read()
            if not ret1:
                break
        else:
            if frame_cnt >= len(image_files):
                break
            frame1 = cv2.imread(image_files[frame_cnt])
            filename = f"IMAGE Mode {os.path.basename(image_files[frame_cnt])}"
            if frame1 is None or frame1.size == 0:
                print(f"Failed to load {image_files[frame_cnt]}, frame_cnt:{frame_cnt}")
                continue

        draw_frame1 = frame1.copy()
        if VIDEO_MODE == 0:
            cv2.putText(draw_frame1, f"frame_cnt {frame_cnt} [{filename}]", (draw_frame1.shape[1] - 500, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        _, frame1 = cv2.threshold(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                cv2.THRESH_TOZERO)

        filtered_blob_area_left = get_blob_area(frame1)
        for blob_id, bbox in enumerate(filtered_blob_area_left):
            (x, y, w, h) = (bbox[2])
            if DO_PYRAMID == 1:
                overlapping = check_blobs_with_pyramid(frame1, draw_frame1, x, y, w, h, max_level)
                # if overlapping == True:
                #     cv2.rectangle(draw_frame1, (x, y), (x + w, y + h), (0, 0, 255), 1, 1)
                #     cv2.putText(draw_frame1, f"SEG", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                #     continue
            # advanced_check_contours(frame1, draw_frame1, (x, y, w, h))
            result_frame = test_code_1(frame1, (x, y, w, h))
            # test_code_2(frame1, draw_frame1)

            cv2.rectangle(draw_frame1, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
            cv2.imshow('result frame', result_frame)
            cv2.imshow('LEFT CAMERA', draw_frame1)
            KEY = cv2.waitKey(0)
            if KEY & 0xFF == 27:
                if VIDEO_MODE == 1:
                    cam_L.release()
                cv2.destroyAllWindows()
                return
            elif KEY == ord('n'):
                if AUTO_LOOP == 0:
                    frame_cnt += 1                   
        

        KEY = cv2.waitKey(1)
        if KEY & 0xFF == 27:
            break
        elif KEY == ord('p'):
            DO_PYRAMID = 1
        elif KEY == ord('n'):
            if AUTO_LOOP == 0:
                frame_cnt += 1                   
        
        if AUTO_LOOP == 1 and VIDEO_MODE == 0:
            frame_cnt += 1

    if VIDEO_MODE == 1:
        cam_L.release()

    cv2.destroyAllWindows()



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

    pyramid_test(camera_devices)

    # image_files = sorted(glob.glob(f"{script_dir}/pyramid_test/*.png"))
    # print(image_files)
    # image = cv2.imread(image_files[5])

    # draw_frame = image.copy()

    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # th, bw = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
     

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


    # borderSize = 75
    # distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize, 
    #                                 cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)

    # gap = 10                                
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(borderSize-gap)+1, 2*(borderSize-gap)+1))
    # kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap, 
    #                                 cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
    

    # distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
    # mn, mx, _, _ = cv2.minMaxLoc(nxcor)
    # th, peaks = cv2.threshold(nxcor, mx*0.5, 255, cv2.THRESH_BINARY)


    # peaks8u = cv2.convertScaleAbs(peaks)
    # contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # peaks8u = cv2.convertScaleAbs(peaks)    # to use as mask

    # for i in range(len(contours)):
    #     x, y, w, h = cv2.boundingRect(contours[i])
    #     _, mx, _, mxloc = cv2.minMaxLoc(dist[y:y+h, x:x+w], peaks8u[y:y+h, x:x+w])
    #     cv2.circle(draw_frame, (int(mxloc[0]+x), int(mxloc[1]+y)), int(mx), (255, 0, 0), 2)
    #     cv2.rectangle(draw_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    #     cv2.drawContours(draw_frame, contours, i, (0, 0, 255), 2)

    # cv2.imshow('result', draw_frame)
    # cv2.waitKey(0)