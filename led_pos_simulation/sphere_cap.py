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

coords = np.array([x_masked, y_masked, z_masked]).T

# 시작점을 기반으로 점 선택
led_coords = select_points(coords, num_leds)

# 3D 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 구 캡을 3D 플롯에 추가
ax.scatter(x_masked, y_masked, z_masked, color='lightgray', marker='o', alpha=0.1)
ax.scatter(led_coords[:, 0], led_coords[:, 1], led_coords[:, 2], color='red', marker='o', s=5)
ax.scatter(coords[0][0], coords[0][1], coords[0][2], color='black', marker='o', s=10)

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

# 새로 시작
def look_at(camera_pos, target_pos):
    z_axis = (camera_pos - target_pos) / np.linalg.norm(camera_pos - target_pos)

    if np.abs(z_axis[0]) < 1e-6 and np.abs(z_axis[1]) < 1e-6:
        x_axis = np.array([1, 0, 0], dtype=np.float32)
    else:
        x_axis = np.array([-z_axis[1], z_axis[0], 0], dtype=np.float32)
        x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    rotation_matrix = np.array([x_axis, y_axis, z_axis])
    translation_vector = -np.matmul(rotation_matrix, camera_pos)

    return rotation_matrix, translation_vector

camera_matrix = np.array([[714.193, 0, 636.242], [0, 714.193, 468.407], [0, 0, 1]])

camera_pos = np.array([30, 0, 0], dtype=np.float32)
origin_pos = np.array([0, 0, 0], dtype=np.float32)

rotation_matrix, translation_vector = look_at(camera_pos, origin_pos)
RT = np.hstack([rotation_matrix, translation_vector.reshape(3, 1)])
rvec, tvec = cv2.decomposeProjectionMatrix(np.matmul(camera_matrix, RT))[:2]
tvec = tvec[:3, 0].reshape((3, 1))  # Reshape the translation vector
projected_points, _ = cv2.projectPoints(led_coords, rvec, tvec, camera_matrix, None)

img = np.zeros((960, 1280, 3), dtype=np.uint8)
for point in projected_points:
    x, y = int(point[0][0]), int(point[0][1])
    cv2.circle(img, (x, y), 3, (255, 255, 255), -1)

cv2.imshow("2D Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
