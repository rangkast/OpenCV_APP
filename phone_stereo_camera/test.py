#!/usr/bin/env python
import threading

import cv2
import numpy as np
import os
import glob
from definition import *

from multiprocessing import Process




def count():
    print('start open_camera')

    cap1 = cv2.VideoCapture(CAM_1[0])

    if not cap1.isOpened():
        sys.exit()

    while True:
        ret1, frame1 = cap1.read()
        if not ret1:
            break

        imgL = frame1.copy()
        cv2.imshow('left camera', imgL)

    cap1.release()
    cv2.destroyAllWindows()
    print('process exit')


if __name__ == '__main__':
    t = threading.Thread(target=count)
    t.start()
    t.join()

    print('main exit')
#
#
# if __name__ == '__main__':
#     # Defining the dimensions of checkerboard
#     CHECKERBOARD = (4, 7)
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#     # Creating vector to store vectors of 3D points for each checkerboard image
#     objpoints = []
#     # Creating vector to store vectors of 2D points for each checkerboard image
#     imgpoints = []
#
#     # Defining the world coordinates for 3D points
#     objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
#     objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
#     prev_img_shape = None
#
#
#     print('start open_camera')
#
#     cap1 = cv2.VideoCapture(CAM_1[0])
#     cap1.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
#     cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)
#
#     if not cap1.isOpened():
#         sys.exit()
#
#     while True:
#         ret1, frame1 = cap1.read()
#         if not ret1:
#             break
#         imgL = frame1.copy()
#         gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
#
#         if cv2.waitKey(1) == ord('c'):
#             ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
#                                                      cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
#
#             """
#             If desired number of corner are detected,
#             we refine the pixel coordinates and display
#             them on the images of checker board
#             """
#             if ret:
#                 objpoints.append(objp)
#                 # refining pixel coordinates for given 2d points.
#                 corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
#
#                 imgpoints.append(corners2)
#
#                 # Draw and display the corners
#                 imgL = cv2.drawChessboardCorners(imgL, CHECKERBOARD, corners2, ret)
#
#         cv2.imshow('left camera', imgL)
#
#     cap1.release()
#     cv2.destroyAllWindows()
#
#     h, w = imgL.shape[:2]
#
#     """
#     Performing camera calibration by
#     passing the value of known 3D points (objpoints)
#     and corresponding pixel coordinates of the
#     detected corners (imgpoints)
#     """
#     ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#
#     print("Camera matrix : \n")
#     print(mtx)
#     print("dist : \n")
#     print(dist)
#     print("rvecs : \n")
#     print(rvecs)
#     print("tvecs : \n")
#     print(tvecs)
#
