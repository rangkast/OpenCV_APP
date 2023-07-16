import numpy as np
import cv2
import os

CV_MIN_THRESHOLD = 150
CV_MAX_THRESHOLD = 255
BLOB_SIZE = 45
def detect_led_lights(image, padding=5, min_area=100, max_area=1000):
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
        area = cv2.contourArea(contour)

        # Check if the area of the contour is within the specified range
        if min_area <= area <= max_area:
            blob_info.append((x, y, w, h))

    return blob_info

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



script_dir = os.path.dirname(os.path.realpath(__file__))
print(os.getcwd())

cap = cv2.VideoCapture('output_rifts_right_9.mkv')

# Get total number of frames
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
current_frame = 0
start_frame = 500

print('total_frames: ', total_frames)

centers1 = []
while(cap.isOpened()):
    # Read frames
    ret, frame = cap.read()

    if ret:
        draw_frame = frame.copy()        
        _, frame = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                cv2.THRESH_TOZERO)
                
        if current_frame == start_frame:
                # LED blob 찾기
            blobs1 = detect_led_lights(frame, 2, 5, 500)
            # 첫 번째 이미지의 LED blob 중심점 계산            
            for blob in blobs1:
                gcx, gcy, gsize = find_center(frame, blob)
                if gsize < BLOB_SIZE:
                    continue
                centers1.append((gcx, gcy, blob))

            # 가장 가까운 이미지와 그 거리를 저장할 변수 초기화
            closest_img = None
            min_distance = float('inf')

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
        blobs2 = detect_led_lights(frame, 2, 5, 500)
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

        # 두 번째 이미지의 중심점들을 이미지에 표시 (빨간색)
        for center in centers2:
            cv2.circle(draw_frame, (int(center[0]), int(center[1])), 1, (0, 0, 255), -1)


        # 두 이미지의 LED 중심점간 거리 계산
        if len(centers1) == len(centers2):
            max_distance = max(np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                            for c1, c2 in zip(centers1, centers2))            
            # 거리가 더 작으면 업데이트
            if current_frame > 100:
                if max_distance < min_distance and current_frame != start_frame:
                    min_distance = max_distance
                    closest_img = current_frame

        # Show the frames
        cv2.imshow('Frame', draw_frame)

        # Wait for a key press and break the loop if 'q' is pressed
        if cv2.waitKey(8) & 0xFF == ord('q'):
            break

        current_frame += 1
    else:
        break

print(f'The closest image to the start image is Image_{closest_img}.')

# Release the video captures and close the windows
cap.release()

cv2.destroyAllWindows()


cap = cv2.VideoCapture('output_rifts_right_9.mkv')

# Get the start_frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Frame indices start from 0
ret1, frame_start = cap.read()

# Get the closest_img frame
cap.set(cv2.CAP_PROP_POS_FRAMES, closest_img)  # Frame indices start from 0
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

    cv2.imshow('Overlapped frame', overlapped_frame)
    cv2.waitKey(0)  # Wait until any key is pressed

cap.release()
cv2.destroyAllWindows()
