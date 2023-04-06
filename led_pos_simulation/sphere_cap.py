import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# 구 캡의 반지름 R과 높이 h
R = 11
UPPER_Z = 2.5
LOWER_Z = -1
num_points = 60
num_leds = 24
camera_matrix = np.array([[712.623, 0., 653.448],
                          [0., 712.623, 475.572],
                          [0., 0., 1.]])

dist_coeff = np.array([[0.072867],
                       [-0.026268],
                       [0.007135],
                       [-0.000997]])


def select_points(coords, num_leds):
    selected_indices = [0]  # 시작점을 첫 번째로 선택

    for _ in range(num_leds - 1):
        min_dists = np.full((coords.shape[0],), np.inf)

        for selected_idx in selected_indices:
            dists = np.linalg.norm(coords - coords[selected_idx], axis=1)
            min_dists = np.minimum(min_dists, dists)

        next_idx = np.argmax(min_dists)
        selected_indices.append(next_idx)

    return coords[selected_indices]


def led_position(ax, R):
    # 구 캡 위의 점들을 계산
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi = np.linspace(0, np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)

    x = R * np.sin(phi) * np.cos(theta)
    y = R * np.sin(phi) * np.sin(theta)
    z = R * np.cos(phi)

    # z 축 기준으로 -1 이하와 2 이상을 자르기
    mask = (z >= LOWER_Z) & (z <= UPPER_Z)
    x_masked = x[mask]
    y_masked = y[mask]
    z_masked = z[mask]
    ax.scatter(x_masked, y_masked, z_masked, color='lightgray', marker='o', alpha=0.1)
    coords = np.array([x_masked, y_masked, z_masked]).T
    # 시작점을 기반으로 점 선택
    led_coords = select_points(coords, num_leds)

    return led_coords


# 3D 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 구 캡을 3D 플롯에 추가
led_coords_o = led_position(ax, R)
ax.scatter(led_coords_o[1:, 0], led_coords_o[1:, 1], led_coords_o[1:, 2], color='red', marker='o', s=5)
ax.scatter(led_coords_o[0][0], led_coords_o[0][1], led_coords_o[0][2], color='black', marker='o', s=10)

# 플롯 옵션 설정
ax.set_title('Spherical Caps with 24 LEDs')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# x, y, z 축 범위 설정
axis_limit = R * 1.5
ax.set_xlim(-axis_limit, axis_limit)
ax.set_ylim(-axis_limit, axis_limit)
ax.set_zlim(-axis_limit, axis_limit)
plt.show()
