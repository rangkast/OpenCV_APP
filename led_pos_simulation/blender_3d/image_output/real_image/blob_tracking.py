import cv2
import numpy as np
import os
import glob

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
            x_sum += x * frame[y][x]
            t_sum += frame[y][x]
            if frame[y][x] >= CV_MIN_THRESHOLD:
                m_count += 1

    for x in range(X, X + W):
        for y in range(Y, Y + H):
            y_sum += y * frame[y][x]

    if t_sum != 0:
        g_c_x = x_sum / t_sum
        g_c_y = y_sum / t_sum

    if g_c_x == 0 or g_c_y == 0:
        return 0, 0, 0

    return g_c_x, g_c_y, m_count

def detect_led_lights(image, padding=5, min_area=100, max_area=1000):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blob_info = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        x -= padding
        y -= padding
        w += padding * 2
        h += padding * 2

        area = cv2.contourArea(contour)

        if min_area <= area <= max_area:
            blob_info.append((x, y, w, h))

    return blob_info

camera_img_path = "./tmp/render/inv/"
CV_MIN_THRESHOLD = 100
CV_MAX_THRESHOLD = 255

script_dir = os.path.dirname(os.path.realpath(__file__))
image_files = sorted(glob.glob(os.path.join(script_dir, camera_img_path + '*.png')))
id_counter = 0  # To count unique IDs for new blobs

frame_id = 0
while True:
    frame = cv2.imread(image_files[frame_id])
    if frame is None or frame.size == 0:
        print(f"Failed to load {image_files[frame_id]}, frame_id:{frame_id}")
        continue
    draw_frame = frame.copy()
    _, frame = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)

    blob_area = detect_led_lights(frame, 2, 5, 500)

    blob_centers = []
    for blob_id, bbox in enumerate(blob_area):
        gcx, gcy, gsize = find_center(frame, bbox)
        if gsize < 45:
            continue

        cv2.rectangle(draw_frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 255, 255), 1, 1)
        blob_centers.append((gcx, gcy, bbox))
        print(f"{blob_id} : {gcx}, {gcy}")

     # Sorting blob_centers by x coordinate of bounding box in ascending order
    blob_centers.sort(key=lambda x: x[2][0])

      # Initialize or update trackers and IDs
    key = cv2.waitKey(1)
    if frame_id == 0 or (key != -1 and 'n' in chr(key).lower()):
        tracker_list = []
        id_list = []
        for i, blob_center in enumerate(blob_centers):
            tracker = cv2.TrackerCSRT_create()
            bbox = blob_center[2]  # Get bounding box from blob_center
            tracker.init(frame, (bbox[0], bbox[1], bbox[2], bbox[3]))
            tracker_list.append(tracker)
            id_list.append(id_counter)
            id_counter += 1  # Increment the id_counter
    else:
        new_tracker_list = []
        new_id_list = []
        is_tracker_added = False  # Check if any tracker is added for the current frame
        for tracker, blob_id, blob_center in zip(tracker_list, id_list, blob_centers):
            ret, bbox = tracker.update(frame)
            if ret:
                new_tracker_list.append(tracker)
                new_id_list.append(blob_id)
                is_tracker_added = True
        if not is_tracker_added:
            # Add new tracker for the new blob
            bbox = blob_centers[-1][2]
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (bbox[0], bbox[1], bbox[2], bbox[3]))
            new_tracker_list.append(tracker)
            new_id_list.append(id_counter)
            id_counter += 1  # Increment the id_counter
        tracker_list = new_tracker_list
        id_list = new_id_list

    # Draw blobs and IDs
    for tracker, blob_id in zip(tracker_list, id_list):
        ret, bbox = tracker.update(frame)
        if ret:
            cv2.rectangle(draw_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (255, 0, 0), 1)
            cv2.putText(draw_frame, f"{blob_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow('Frame', draw_frame)
    key = cv2.waitKey(0)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('n'):
        frame_id += 1
    elif key & 0xFF == ord('b') and frame_id > 0:
        frame_id -= 1

cv2.destroyAllWindows()
