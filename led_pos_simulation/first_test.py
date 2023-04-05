import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 평행한 두 원의 반지름 R과 거리 D
R = 5
D = 1

# 작은 원들의 개수와 각도
num_circles_along_radius = 8
num_circles_parallel = 2
angle_along_radius = 360 / num_circles_along_radius
angle_parallel = 360 / num_circles_parallel

# 작은 원들의 3D 좌표를 계산하는 함수
def calculate_coordinates(R, D, theta, phi):
    x = R * np.cos(np.radians(theta))
    y = R * np.sin(np.radians(theta))
    z = D * (phi / 360)
    return x, y, z

# 작은 원들의 좌표를 저장할 리스트
circle_coordinates = []

# 각 원에 대한 좌표 계산
for i in range(num_circles_along_radius):
    for j in range(num_circles_parallel):
        theta = i * angle_along_radius
        phi = j * angle_parallel
        x, y, z = calculate_coordinates(R, D, theta, phi)
        circle_coordinates.append((x, y, z))

# 3D 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 작은 원들을 3D 플롯에 추가
for coord in circle_coordinates:
    ax.scatter(coord[0], coord[1], coord[2])

# 플롯 옵션 설정
ax.set_title('Small Circles on Parallel Rings')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 결과 표시
plt.show()
