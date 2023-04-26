import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def plot_object(ax, object_points):
    ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], c='r', marker='o')

def set_camera_position(ax, position, roll, pitch, yaw):
    # Create a rotation matrix from roll, pitch, yaw angles
    rotation = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    rotation_matrix = rotation.as_matrix()

    # Set camera position and orientation
    ax.view_init(elev=-pitch, azim=-yaw)
    ax.set_proj_type('persp')

# Object points (example: a cube)
object_points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
])

# Camera position and orientation (example values)
camera_position = np.array([10, 20, 30])
roll, pitch, yaw = 30, 40, 50

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the object and set the camera position
plot_object(ax, object_points)
set_camera_position(ax, camera_position, roll, pitch, yaw)

# Display the plot
plt.show()
