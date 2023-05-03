import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from function import *

def sphere_coordinates(theta, phi, r):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def objective_function(params, points):
    cx, cy, cz, r = params
    center = np.array([cx, cy, cz])
    error = 0
    for point in points:
        dist = np.linalg.norm(center - point)
        error += (dist - r) ** 2
    return error

def point_to_sphere_distance(point, sphere_center, sphere_radius):
    distance_to_center = np.linalg.norm(point - sphere_center)
    distance_to_surface = abs(distance_to_center - sphere_radius)
    return distance_to_surface


# 초기 추정값 설정
initial_center = np.mean(points, axis=0)
initial_radius = np.linalg.norm(points[0] - initial_center)
initial_guess = [*initial_center, initial_radius]

result = minimize(objective_function, initial_guess, args=(points,), method='Nelder-Mead')

center_optimal, radius_optimal = result.x[:3], result.x[3]
print(f"Optimal center: {center_optimal}, Optimal radius: {radius_optimal}")

for i, leds in enumerate(points):
    dis_to_surface = point_to_sphere_distance(leds, center_optimal, radius_optimal)
    print(f"idx {i}, distance {dis_to_surface}")


# 그래프 그리기
fig = plt.figure(figsize=(10, 10), num='Sphere Simulator')
ax = fig.add_subplot(111, projection="3d")

# 점들을 파란색으로 그리기
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="b", marker="o")

# 추정된 구를 연한 회색으로 그리기
theta, phi = np.linspace(0, np.pi, 100), np.linspace(0, 2 * np.pi, 100)
THETA, PHI = np.meshgrid(theta, phi)
sphere_coords = sphere_coordinates(THETA, PHI, radius_optimal) + center_optimal[:, np.newaxis, np.newaxis]
ax.plot_surface(sphere_coords[0], sphere_coords[1], sphere_coords[2], color="lightgray", alpha=0.5)

ax.set_xlim([-0.3, 0.3])
ax.set_xlabel('X')
ax.set_ylim([-0.3, 0.3])
ax.set_ylabel('Y')
ax.set_zlim([-0.3, 0.3])
ax.set_zlabel('Z')

scale = 1.5
f = zoom_factory(ax, base_scale=scale)

# 텍스트 레이블 추가
ax.text(points[0, 0], points[0, 1], points[0, 2], "Points", color="blue")
ax.text(sphere_coords[0, 0, 0], sphere_coords[1, 0, 0], sphere_coords[2, 0, 0], "Estimated Sphere", color="gray")

plt.show()
