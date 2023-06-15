import numpy as np
import cv2
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

target_pose_led_data = np.array([
     [-0.01108217, -0.00278021, -0.01373098],
     [-0.02405356,  0.00777868, -0.0116913 ],
     [-0.02471722,  0.00820648,  0.00643996],
     [-0.03312733,  0.02861635,  0.0013793 ],
     [-0.02980387,  0.03374299,  0.02675826],
     [-0.0184596,  0.06012725,  0.02233215],
     [-0.00094422,  0.06020401,  0.04113377],
     [ 0.02112556,  0.06993855,  0.0256014 ],
     [ 0.04377158,  0.05148328,  0.03189337],
     [ 0.04753083,  0.05121397,  0.01196245],
     [ 0.0533449,  0.02829823, 0.01349697],
     [ 0.05101214,  0.02247323, -0.00647229],
     [ 0.04192879,  0.00376628, -0.00139432],
     [ 0.03947314,  0.00479058, -0.01699771],
     [ 0.02783124, -0.00088511, -0.01754906],
])

image_coordinates = np.array([
    [657.4802894356005, 504.83178002894357],
    [631.7905127712694, 500.90293835221814],
    [630.633866114346, 459.46895150259877],
    [581.3196922716902, 470.61686070321184]
])

camera_matrix = np.array([[712.623, 0.0, 653.448],
                          [0.0, 712.623, 475.572],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
dist_coeffs = np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)

# Initial estimate of camera pose (could also be calculated via solvePnP, for example)
rvec = np.array([0.0, 0.0, 0.0])  # rotation vector
tvec = np.array([0.0, 0.0, 0.0])  # translation vector

# Project 3D points to 2D
projected_2D_points = cv2.projectPoints(target_pose_led_data, rvec, tvec, camera_matrix, dist_coeffs)[0].reshape(-1, 2)

# Create distance matrix between projected 2D points and measured 2D points
dist_matrix = distance_matrix(projected_2D_points, image_coordinates)

# Perform assignment to find optimal match
row_ind, col_ind = linear_sum_assignment(dist_matrix)

for led_num, img_coord in zip(row_ind, image_coordinates[col_ind]):
    print(f"LED 번호 {led_num + 1}는 2D 이미지 좌표 {img_coord}에 매칭됩니다.")
