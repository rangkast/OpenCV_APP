import numpy as np
# 사용 예시
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from function import *


def cylinder_select_points(coords, num_leds, led_r,  min_start_distance=0):
    if num_leds > coords.shape[0]:
        raise ValueError("num_leds cannot be greater than the number of points in coords")

    selected_indices = [0]  # 시작점을 첫 번째로 선택

    for _ in range(num_leds - 1):  # 시작점이 이미 선택되었으므로 num_leds - 1
        min_dists = np.full((coords.shape[0],), np.inf)

        for selected_idx in selected_indices:
            dists = np.linalg.norm(coords - coords[selected_idx], axis=1) + led_r * 2
            min_dists = np.minimum(min_dists, dists)

        # 시작점과의 최소 거리 조건을 적용
        min_dists[:1] = 0  # 시작점 자체의 거리를 0으로 설정
        valid_indices = np.where(min_dists >= min_start_distance)

        if valid_indices[0].size == 0:  # 모든 점이 거리 조건을 충족하지 않는 경우
            return ERROR

        next_idx = np.argmax(min_dists[valid_indices])
        selected_indices.append(valid_indices[0][next_idx])

    return coords[selected_indices]


def create_cylindrical_surface_coords(radius, center, lower_z, upper_z, padding, num_points=100, draw=False, ax=None):
    theta = np.linspace(0, 2 * np.pi, num_points)

    # 원기둥의 높이를 계산하고 이에 따른 z 좌표 생성
    z = np.linspace(lower_z - padding, upper_z + padding, num_points)
    theta, z = np.meshgrid(theta, z)

    # 원기둥 표면 좌표 계산
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    mask = (z >= lower_z) & (z <= upper_z)
    x_masked = x[mask]
    y_masked = y[mask]
    z_masked = z[mask]

    coords = np.vstack([x_masked.ravel(), y_masked.ravel(), z_masked.ravel()]).T

    # 위아래 패딩 좌표 추가
    padding_mask = ((z >= lower_z - padding) & (z <= lower_z)) | ((z >= upper_z) & (z <= upper_z + padding))
    x_masked = x[padding_mask]
    y_masked = y[padding_mask]
    z_masked = z[padding_mask]
    ax.scatter(x_masked, y_masked, z_masked, color='green', marker='.', alpha=0.1)

    padding_coords = np.vstack([x_masked.ravel(), y_masked.ravel(), z_masked.ravel()]).T
    combined_coords = np.vstack((coords, padding_coords))

    if draw and ax is not None:
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], color='lightgray', marker='.', alpha=0.1)

    return combined_coords, coords


radius = 0.05
center = [0, 0]
lower_z = -0.025
upper_z = 0.025
num_leds = 15
led_r = 0.003
padding = led_r * 2 + 0.005

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
combined_coords, coords = create_cylindrical_surface_coords(radius, center, lower_z, upper_z, padding,  num_points=100, draw=True, ax=ax)

ret_coords = cylinder_select_points(coords, num_leds, led_r)

print('ret_coords', ret_coords)
# ax.scatter(ret_coords[0][0], ret_coords[0][1], ret_coords[0][2], color='black',
#            marker='o', s=5)
# ax.text(ret_coords[0][0], ret_coords[0][1], ret_coords[0][2], 'S', color='black', fontsize=10)

# 그 외 점
for i in range(0, num_leds):
    ax.scatter(ret_coords[i, 0], ret_coords[i, 1], ret_coords[i, 2], color='red',
               marker='o', s=5)
    ax.text(ret_coords[i, 0], ret_coords[i, 1], ret_coords[i, 2], str(i), color='black', fontsize=7)


# 원점
ax.scatter(0, 0, 0, marker='o', color='k', s=10)
ax.set_xlim([-0.3, 0.3])
ax.set_xlabel('X')
ax.set_ylim([-0.3, 0.3])
ax.set_ylabel('Y')
ax.set_zlim([-0.3, 0.3])
ax.set_zlabel('Z')
scale = 1.5

f = zoom_factory(ax, base_scale=scale)

plt.show()

file = './result_cylinder.pickle'
data = OrderedDict()

data['LED_INFO'] = ret_coords
data['MODEL_INFO'] = combined_coords
ret = pickle_data(WRITE, file, data)
if ret != ERROR:
    print('data saved')

