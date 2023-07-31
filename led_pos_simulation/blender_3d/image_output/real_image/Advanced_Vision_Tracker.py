import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from dlib_tracker import BlobTracker  # BlobTracker를 정의한 코드 파일
from Advanced_Function import *

def track_led_lights(camera_img_path, tracker):
    # Load all images in the directory
    images = sorted(glob.glob(camera_img_path + "*.png"))  # 이미지 파일 확장자가 jpg가 아닌 경우 수정 필요

    # To keep track of blob colors
    blob_colors = {}

    for image_path in images:
        image = cv2.imread(image_path, 0)
        blobs = detect_led_lights(image, padding=5, min_area=100, max_area=1000)
        tracker.update(blobs)

        # Draw blobs and tracking paths
        for idx, blob in enumerate(tracker.get_tracked_blobs()):
            bbox, blob_id = blob
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Assign a color to the blob id if it doesn't have one
            if blob_id not in blob_colors:
                blob_colors[blob_id] = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

            # Draw tracking path
            cv2.circle(image, (x+w//2, y+h//2), 1, blob_colors[blob_id], -1)

        # Show the image
        cv2.imshow("Image", image)
        cv2.waitKey(100)  # Display each image for 100ms

    cv2.destroyAllWindows()

# Initialize the blob tracker
blob_tracker = BlobTracker()
camera_img_path = f"./tmp/render/ARCTURAS/plane/"
# Start tracking the LED lights in the images
track_led_lights(script_dir + camera_img_path, blob_tracker)
