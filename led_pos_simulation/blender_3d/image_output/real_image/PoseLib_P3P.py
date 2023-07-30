import sys
import os
import numpy as np

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Add the directory containing poselib to the module search path
print(script_dir)
sys.path.append(os.path.join(script_dir, '../../../../EXTERNALS'))

import poselib
# help(poselib.p3p)



origin_led_data = np.array([
    [-0.02146761, -0.00343424, -0.01381839],
    [-0.0318701, 0.00568587, -0.01206734],
    [-0.03692925, 0.00930785, 0.00321071],
    [-0.04287211, 0.02691347, -0.00194137],
    [-0.04170018, 0.03609551, 0.01989264],
    [-0.02923584, 0.06186962, 0.0161972],
    [-0.01456789, 0.06295633, 0.03659283],
    [0.00766914, 0.07115411, 0.0206431],
    [0.02992447, 0.05507271, 0.03108736],
    [0.03724313, 0.05268665, 0.01100446],
    [0.04265723, 0.03016438, 0.01624689],
    [0.04222733, 0.0228845, -0.00394005],
    [0.03300807, 0.00371497, 0.00026865],
    [0.03006234, 0.00378822, -0.01297127],
    [0.02000199, -0.00388647, -0.014973]
])

image_coordinates = np.array(
[[-0.04967391,  0.03472004],
       [-0.08161921,  0.02982381],
       [-0.09622432, -0.00874314]]
)
camera_matrix = [
    [np.array([[712.623, 0.0, 653.448],
               [0.0, 712.623, 475.572],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)],
]
default_cameraK = np.eye(3).astype(np.float64)
from itertools import permutations

# origin_led_data와 image_coordinates 정의는 생략

# 모든 가능한 3D-2D 점 페어를 찾음
pairs = list(permutations(range(len(origin_led_data)), 3))  # poselib.p3p requires exactly 3 pairs

min_error = np.inf
best_pair = None
best_pose = None



# def is_valid_pose(pose):
#     if pose is None:
#         return False
#     q, t = pose.q, pose.t
#     return not np.isnan(q).any() and not np.isnan(t).any()

# def calculate_pose(p3d, p2d):
#     # Convert 2D points to homogeneous coordinates
#     p2d_homogeneous = np.hstack((p2d, np.ones((p2d.shape[0], 1))))

#     # poselib.p3p requires 3 points to solve pose
#     poses = poselib.p3p(p2d_homogeneous, p3d)
#     for pose in poses:
#         if is_valid_pose(pose):
#             return pose
#     return None

# from scipy.spatial.transform import Rotation as R

# def project_3d_to_2d(p3d, pose):
#     # Ensure p3d is a 2D array
#     p3d = np.atleast_2d(p3d)
    
#     # Convert quaternion to rotation matrix
#     rotation_matrix = R.from_quat(pose.q).as_matrix()

#     # Use camera matrix to project 3D points to 2D
#     # K = camera_matrix[0][0]  # Assuming camera_matrix is defined
#     K = default_cameraK  # Assuming camera_matrix is defined
#     projected = np.dot(K, np.dot(rotation_matrix, p3d.T) + pose.t.reshape(-1, 1))
    
#     return (projected[:-1] / projected[-1]).T


# # 각 페어에 대해
# for pair in pairs:
#     # 대응되는 3D 점과 2D 점을 선택
#     selected_3d_points = origin_led_data[list(pair)]
#     selected_2d_points = image_coordinates

#     # pose를 계산 (여기서는 P3P 알고리즘을 사용)
#     pose = calculate_pose(selected_3d_points, selected_2d_points)

#     # If no valid pose found, skip this pair
#     if pose is None:
#         continue

#     # pose를 이용해 3D 점을 2D로 투영
#     projected_2d_points = project_3d_to_2d(selected_3d_points, pose)

#     # 재투영 오차를 계산
#     error = np.sum((selected_2d_points - projected_2d_points)**2)

#     # 오차가 최소인 페어와 pose를 저장
#     if error < min_error:
#         min_error = error
#         best_pair = pairq
#         best_pose = pose

# print("Best pair:", best_pair)
# print("Best pose:", best_pose)

# help(poselib)
import numpy as np
import cv2

# 카메라 매트릭스와 왜곡 계수
camera_matrix = np.array([[712.623, 0.0, 653.448],
                          [0.0, 712.623, 475.572],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
dist_coeffs = np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)

# 이미지 좌표
points2D = np.array([[572.21915805, 502.43690588],
                     [619.6828693, 464.52776489],
                     [662.28616424, 498.58342082]])

# 이미지 좌표를 3D로 확장
points2D_EX = np.expand_dims(points2D, axis=1)

# 좌표 정규화
normalized_points = cv2.undistortPoints(points2D_EX, camera_matrix, dist_coeffs)
# 홈젠리우스 좌표계로 변환 ([u, v] -> [u, v, 1])
normalized_points = np.concatenate([normalized_points, np.ones((normalized_points.shape[0], 1, 1))], axis=-1)
print(normalized_points)

for data in points2D:
    print(f"{(data[0] - camera_matrix[0][2]) / camera_matrix[0][0]},{(data[1] - camera_matrix[1][2]) / camera_matrix[1][1]}")


# help(poselib)

import math
total_cases = 25 * math.comb(25, 4) * math.perm(25, 4)
print(total_cases)
