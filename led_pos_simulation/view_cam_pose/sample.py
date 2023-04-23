import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 예제 카메라 행렬 및 거리 계수
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
dist_coeffs = np.zeros((4, 1))

# 3D 월드 좌표 (마이너스 위치값 추가)
object_points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
    [-1, -1, 0],
    [-1, 0, -1],
], dtype=np.float32)

# 2D 이미지 좌표
image_points = np.array([
    [320, 240],
    [420, 240],
    [320, 340],
    [320, 140],
    [420, 340],
    [420, 140],
    [220, 240],
    [320, 140],
    [320, 340],
    [220, 140],
    [220, 340],
], dtype=np.float32)
# 그래프 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# solvePnP를 사용하여 R|T 구하기
_, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

# 회전 벡터를 회전 행렬로 변환
R, _ = cv2.Rodrigues(rvec)

# 카메라 위치 계산
camera_position = -R.T @ tvec

# 카메라 방향 계산
camera_direction = R.T @ np.array([[0, 0, -1]]).T
camera_direction = camera_direction.ravel()

# 카메라 optical axis 방향 벡터 계산
optical_axis = R.T @ np.array([0, 0, -1])

# 카메라 위치에서 optical axis까지의 방향 벡터 계산
direction_vector = optical_axis - camera_position.ravel()

ax.quiver(
    camera_position[0], camera_position[1], camera_position[2],
    direction_vector[0], direction_vector[1], direction_vector[2],
    color="blue", label="Camera Direction", length=1, normalize=True
)




# 카메라 위치 표시 (파란색 점)
ax.scatter(camera_position[0], camera_position[1], camera_position[2], color="blue", label="Camera Position")

# 카메라 방향 표시 (빨간색 화살표)
ax.quiver(
    camera_position[0], camera_position[1], camera_position[2],
    camera_direction[0], camera_direction[1], camera_direction[2],
    color="red", label="Camera Direction", length=1, normalize=True
)

# 3D object points 표시 (녹색 점)
ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], color="green", label="Object Points")

# 축 레이블 및 범례 추가
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

plt.show()
