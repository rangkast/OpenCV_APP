from function import *
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ellipsoid_coordinates(theta, phi, a, b, c):
    x = a * np.sin(theta) * np.cos(phi)
    y = b * np.sin(theta) * np.sin(phi)
    z = c * np.cos(theta)
    return np.array([x, y, z])

def objective_function(params, points):
    cx, cy, cz, a, b, c = params
    center = np.array([cx, cy, cz])
    error = 0
    for point in points:
        transformed_point = (point - center) / np.array([a, b, c])
        error += (np.linalg.norm(transformed_point) - 1) ** 2
    return error

def point_to_ellipsoid_distance(point, ellipsoid_center, ellipsoid_radii):
    a, b, c = ellipsoid_radii
    distance_to_center = np.linalg.norm((point - ellipsoid_center) / ellipsoid_radii)
    distance_to_surface = abs(distance_to_center - 1) * np.min(ellipsoid_radii)
    return distance_to_surface

# 초기 추정값 설정
initial_center = np.array([0, 0, 0])
initial_radii = np.max(np.linalg.norm(points, axis=1))
initial_guess = [*initial_center, initial_radii, initial_radii, initial_radii]


result = minimize(objective_function, initial_guess, args=(points,), method='Nelder-Mead')

center_optimal, radii_optimal = result.x[:3], result.x[3:]
print(f"Optimal center: {center_optimal}, Optimal radii: {radii_optimal}")

for i, point in enumerate(points):
    dis_to_surface = point_to_ellipsoid_distance(point, center_optimal, radii_optimal)
    print(f"idx {i}, distance {dis_to_surface}")

# 그래프 그리기
fig = plt.figure(figsize=(10, 10), num='Ellipsoid Simulator')
ax = fig.add_subplot(111, projection="3d")

# 점들을 파란색으로 그리기
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="b", marker="o")

# 추정된 타원을 연한 회색으로 그리기
theta, phi = np.linspace(0, np.pi, 100), np.linspace(0, 2 * np.pi, 100)
THETA, PHI = np.meshgrid(theta, phi)
ellipsoid_coords = ellipsoid_coordinates(THETA, PHI, *radii_optimal) + center_optimal[:, np.newaxis, np.newaxis]
ax.plot_surface(ellipsoid_coords[0], ellipsoid_coords[1], ellipsoid_coords[2], color="lightgray", alpha=0.5)

ax.set_xlim([-0.5, 0.5])
ax.set_xlabel('X')
ax.set_ylim([-0.5, 0.5])
ax.set_ylabel('Y')
ax.set_zlim([-0.5, 0.5])
ax.set_zlabel('Z')

scale = 1.5
f = zoom_factory(ax, base_scale=scale)

# 텍스트 레이블 추가
ax.text(points[0, 0], points[0, 1], points[0, 2], "Points", color="blue")
ax.text(ellipsoid_coords[0, 0, 0], ellipsoid_coords[1, 0, 0], ellipsoid_coords[2, 0, 0], "Estimated Ellipsoid", color="gray")

plt.show()
