import numpy as np
import cv2

CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
CV_MIN_THRESHOLD = 100
CV_MAX_THRESHOLD = 255

# Read the two videos
cap_0 = cv2.VideoCapture('./image_output/real_image/camera_0.mkv')
cap_1 = cv2.VideoCapture('./image_output/real_image/camera_1.mkv')


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
        blob_info.append([x, y, w, h])
    return blob_info


while True:
    # Read frames
    ret_0, frame_0 = cap_0.read()
    ret_1, frame_1 = cap_1.read()
    # Break the loop if one of the videos is over
    if not ret_0 or not ret_1:
        break

    _, frame_0 = cv2.threshold(cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                               cv2.THRESH_TOZERO)
    _, frame_1 = cv2.threshold(cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                               cv2.THRESH_TOZERO)
    blob_area_0 = detect_led_lights(frame_0, 5)
    blob_area_1 = detect_led_lights(frame_1, 5)
    draw_frame_0 = frame_0.copy()
    draw_frame_1 = frame_1.copy()

    # Draw bounding boxes on each frame
    for x, y, w, h in blob_area_0:
        cv2.rectangle(draw_frame_0, (x, y), (x + w, y + h), (255, 255, 255), 2)
    for x, y, w, h in blob_area_1:
        cv2.rectangle(draw_frame_1, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Stack the frames horizontally
    STACK_FRAME = np.hstack((draw_frame_0, draw_frame_1))

    # Add labels with reduced line thickness
    cv2.putText(STACK_FRAME, 'LEFT', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 0, cv2.LINE_AA)
    cv2.putText(STACK_FRAME, 'RIGHT', (frame_0.shape[1] + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 0,
                cv2.LINE_AA)

    # Show the frames
    cv2.imshow('STACK Frame', STACK_FRAME)

    # Wait for a key press and break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video captures and close the windows
cap_0.release()
cap_1.release()
cv2.destroyAllWindows()
