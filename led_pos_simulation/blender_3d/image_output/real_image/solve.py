from triangulate import *
cam_id = 0
points2D = np.array([[725.2362319480634, 470.25884379278364], [705.2640623105831, 477.41474724208996], [679.0872985099429, 449.36708580291366], [551.708309301342, 483.34994424577997]], dtype=np.float64)
points2d_U = np.array(
    cv2.undistortPoints(points2D, camera_matrix[cam_id][0], camera_matrix[cam_id][1])).reshape(-1, 2)
LED_NUM = np.array([0, 1, 2, 5])
points3D = []
for IDX in LED_NUM:
    points3D.append(origin_led_data[IDX])
points3D = np.array(points3D, dtype=np.float64)
print('point_3d\n', points3D)
print('point_2d\n', points2D)
INPUT_ARRAY = [
    cam_id,
    points3D,
    points2D if undistort == 0 else points2d_U,
    camera_matrix[cam_id][0] if undistort == 0 else default_cameraK,
    camera_matrix[cam_id][1] if undistort == 0 else default_dist_coeffs
]
METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P
ret, rvec, tvec, inliers = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)

print('rvec', rvec)
print('tvec', tvec)