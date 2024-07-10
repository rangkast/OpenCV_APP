import matplotlib.pyplot as plt

from function import *
from definition import *

import cv2

num_points = 150
radius = 0.05
led_info = np.array([[0.05, 0, 0]])
lower_z = -radius
upper_z = radius

theta = np.linspace(0, 2 * np.pi, num_points)
phi = np.linspace(0, np.pi, num_points)
theta, phi = np.meshgrid(theta, phi)

x = radius * np.sin(phi) * np.cos(theta)
y = radius * np.sin(phi) * np.sin(theta)
z = radius * np.cos(phi)

mask = (z >= lower_z) & (z <= upper_z)
x_masked = x[mask]
y_masked = y[mask]
z_masked = z[mask]

coords = np.vstack([x_masked.ravel(), y_masked.ravel(), z_masked.ravel()]).T


print('coords', coords)
print('led_info', led_info)
plt.style.use('default')
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, color='lightgray', marker='.', alpha=0.1)

ax.scatter(0, 0, 0, marker='o', color='k', s=20)

ax.scatter(led_info[:, 0], led_info[:, 1], led_info[:, 2], color='red', marker='o', s=20)

ax.set_xlim([-0.2, 0.2])
ax.set_xlabel('X')
ax.set_ylim([-0.2, 0.2])
ax.set_ylabel('Y')
ax.set_zlim([-0.2, 0.2])
ax.set_zlabel('Z')
scale = 1.5
f = zoom_factory(ax, base_scale=scale)


file = './basic_test.pickle'
data = OrderedDict()

data['LED_INFO'] = led_info
data['MODEL_INFO'] = coords

ret = pickle_data(WRITE, file, data)
if ret != ERROR:
    print('data saved')


plt.show()
