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
VIDEO_MODE = 0
CV_MAX_THRESHOLD = 255
CV_MIN_THRESHOLD = 100
BLOB_SIZE = 10
TRACKER_PADDING = 5
max_level = 1
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
    c0 = contours[0]
    AREA = cv2.contourArea(c0)

    print(f"contour area: {AREA}")


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
            # save_image(img_with_contours, f"debug_{i}_{FOUND_STATUS}")
            # save_image(draw_frame_with_shapes, f"debug_draw_{i}_{FOUND_STATUS}")
        
    return FOUND_STATUS


def advanced_check_contours(image, draw_frame):
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
        cv2.putText(draw_frame1, f"frame_cnt {frame_cnt} [{filename}]", (draw_frame1.shape[1] - 500, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        _, frame1 = cv2.threshold(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                cv2.THRESH_TOZERO)

        filtered_blob_area_left = get_blob_area(frame1)
        for blob_id, bbox in enumerate(filtered_blob_area_left):
            (x, y, w, h) = (bbox[2])
            if DO_PYRAMID == 1:
                overlapping = check_blobs_with_pyramid(frame1, draw_frame1, x, y, w, h, max_level)
                if overlapping == True:
                    cv2.rectangle(draw_frame1, (x, y), (x + w, y + h), (0, 0, 255), 1, 1)
                    cv2.putText(draw_frame1, f"SEG", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    continue

            # TEST CODE
            advanced_check_contours(frame1, draw_frame1)

            cv2.rectangle(draw_frame1, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)

        cv2.imshow('LEFT CAMERA', draw_frame1)

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
