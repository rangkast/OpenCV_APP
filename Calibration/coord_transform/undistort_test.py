import numpy as np
import cv2

DIM = (1280, 960)
K = np.array([[712.623, 0.0, 653.448],
              [0.0, 712.623, 475.572],
              [0.0, 0.0, 1.0]], dtype=np.float32)
D = np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float32)
img = np.zeros((720, 953, 3), dtype=np.uint8)
img = cv2.rectangle(img, (200, 150), (300, 200), (255, 255, 255), -1)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

# print('map 1 ', map1)
# print('map 2 ', map2)
undistorted_img = cv2.remap(
    img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
)

# The rectangle is now found at 60, 43 in undistorted_img.  This code works.
new_inlier = []
new_inlier.append([[200, 150]])

new_inlier = np.array(new_inlier, dtype=np.float32)
print(new_inlier)
# rvecs = [np.zeros((1, 10, 3), dtype=np.float32)]
# rvecs[0][0][1][2] = 1
# print(rvecs)

# test_data = [[[200, 150], [210, 133]]]
#
# test_data = np.array(test_data, dtype=np.float32)
# print(test_data)
un_2d = cv2.fisheye.undistortPoints(new_inlier, K, D)
print(un_2d)

d_2d = cv2.fisheye.distortPoints(un_2d, K, D)
# returns array([[[-1.0488918, -0.8601203]]], dtype=float32)
print(d_2d)

# un_2d = cv2.undistortPoints(np.array([[[200, 150]]], dtype=np.float32), K, D)
# print(un_2d)
#
# d_2d = cv2.fisheye.distortPoints(un_2d, K, D)
# # returns array([[[-1.0488918, -0.8601203]]], dtype=float32)
# print(d_2d)
#
# # returns array([[[1064.9419,  822.5983]]], dtype=float32)
# print(cv2.fisheye.undistortPoints(np.array([[[60, 34]]], dtype=np.float32), K, D))
# # Returns array([[[-4.061374 , -3.3357866]]], dtype=float32)
# print(cv2.fisheye.distortPoints(np.array([[[60, 34]]], dtype=np.float32), K, D))
# # array([[[1103.0706 ,  738.13654]]], dtype=float32)
