import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import deque
from itertools import product
from scipy.spatial.distance import pdist, squareform
from function import *


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

combined_coords, coords = create_cylindrical_surface_coords(radius, center, lower_z, upper_z, padding, num_points=100,
                                                            draw=True, ax=ax)

spacing = 0.02
theta = np.arange(0, 2 * np.pi, spacing / radius)
z = np.arange(lower_z, upper_z, spacing)

# Generate the theta and z values
theta = np.arange(0, 2 * np.pi, spacing / radius)
z = np.arange(lower_z, upper_z, spacing)

# Placeholder for the points
all_points = []

# Iterate through all the theta and z values
for t in theta:
    for z_val in z:
        # Calculate the x and y values
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        # If x > 0, add it to the list of points
        if x > 0:
            all_points.append([x, y, z_val])

# Convert to a numpy array
all_points = np.array(all_points)

# Set the seed for reproducibility
np.random.seed(4)

# Randomly select 6 points
selected_indices = np.random.choice(len(all_points), 6, replace=False)
selected_points = all_points[selected_indices]

ax.scatter(selected_points[:, 0], selected_points[:, 1], selected_points[:, 2], marker="o", color="r", alpha=0.7)

# 원점
ax.scatter(0, 0, 0, marker='o', color='k', s=10)
ax.set_xlim([-0.2, 0.2])
ax.set_xlabel('X')
ax.set_ylim([-0.2, 0.2])
ax.set_ylabel('Y')
ax.set_zlim([-0.2, 0.2])
ax.set_zlabel('Z')
scale = 1.5

f = zoom_factory(ax, base_scale=scale)

plt.show()

file = './result_cylinder_base.pickle'
data = OrderedDict()


p_coords_changed = []

for i, coord in enumerate(selected_points):
    # LED 오브젝트의 위치를 조정합니다.
    normalized_direction = Vector(coord).normalized()
    distance_to_o = led_r * 0
    p_coords_changed.append([round(coord[0] - distance_to_o * normalized_direction[0], 8),
                             round(coord[1] - distance_to_o * normalized_direction[1], 8),
                             round(coord[2], 8)])

data['LED_INFO'] = p_coords_changed
data['MODEL_INFO'] = combined_coords
ret = pickle_data(WRITE, file, data)
if ret != ERROR:
    print('data saved')
