from function import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sphere_coordinates(theta, phi, r):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

# 구 A와 구 B의 중심점과 반지름
center_A = np.array([8.44486960e-05, 4.98382521e-03, 5.28879497e-02])
radius_A = 0.06193582267882704
center_B = np.array([-9.41644937e-05, 3.28646054e-02, 2.63934449e-04])
radius_B = 0.04385866975905209

# 그래프 그리기
fig = plt.figure(figsize=(10, 10), num='Sphere Simulator')
ax = fig.add_subplot(111, projection="3d")

# 접선 기준으로 각 구를 그리기
theta_A = np.linspace(0, np.pi / 2, 50)
theta_B = np.linspace(np.pi / 2, np.pi, 50)
phi = np.linspace(0, 2 * np.pi, 100)
THETA_A, PHI_A = np.meshgrid(theta_A, phi)
THETA_B, PHI_B = np.meshgrid(theta_B, phi)

sphere_coords_A = sphere_coordinates(THETA_A, PHI_A, radius_A) + center_A[:, np.newaxis, np.newaxis]
sphere_coords_B = sphere_coordinates(THETA_B, PHI_B, radius_B) + center_B[:, np.newaxis, np.newaxis]

ax.plot_surface(sphere_coords_A[0], sphere_coords_A[1], sphere_coords_A[2], color="lightgray", alpha=0.5)
ax.plot_surface(sphere_coords_B[0], sphere_coords_B[1], sphere_coords_B[2], color="lightgray", alpha=0.5)

# 원점 표시
ax.scatter(0, 0, 0, c="black", marker="o", s=50, label="Origin")
ax.text(0, 0, 0, "Origin", color="black")

# 구의 접면 표시
tangent_point = (center_A + center_B) / 2
ax.scatter(tangent_point[0], tangent_point[1], tangent_point[2], c="green", marker="o", s=50, label="Tangent Point")
ax.text(tangent_point[0], tangent_point[1], tangent_point[2], "Tangent Point", color="green")

ax.set_xlim([-0.1, 0.1])
ax.set_xlabel('X')
ax.set_ylim([-0.1, 0.1])
ax.set_ylabel('Y')
ax.set_zlim([-0.1, 0.1])
ax.set_zlabel('Z')

plt.legend()
plt.show()
