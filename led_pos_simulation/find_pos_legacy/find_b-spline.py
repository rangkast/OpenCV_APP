from function import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def ellipsoid_coordinates(theta, phi, r1, r2):
    z_ratio = np.cos(theta)
    x = r1 * (1 + z_ratio) * np.sin(theta) * np.cos(phi) + r2 * (1 - z_ratio) * np.sin(theta) * np.cos(phi)
    y = r1 * (1 + z_ratio) * np.sin(theta) * np.sin(phi) + r2 * (1 - z_ratio) * np.sin(theta) * np.sin(phi)
    z = (r1 - r2) / 2 * z_ratio
    return np.array([x, y, z])

r1 = 0.0619358
r2 = 0.0438586

theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2 * np.pi, 100)

THETA, PHI = np.meshgrid(theta, phi)

ellipsoid_coords = ellipsoid_coordinates(THETA, PHI, r1, r2)

lower_z = -0.2  # 아래쪽 자르는 값
upper_z = 0.2   # 위쪽 자르는 값

mask = (ellipsoid_coords[2] >= lower_z) & (ellipsoid_coords[2] <= upper_z)
mask = np.tile(mask, (3, 1, 1))

masked_coords = np.where(mask, ellipsoid_coords, np.nan)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(masked_coords[0], masked_coords[1], masked_coords[2], color='lightgray', alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

scale = 1.5
f = zoom_factory(ax, base_scale=scale)

plt.show()
