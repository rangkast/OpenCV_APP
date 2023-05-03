from vector_data import *
from function import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import TextBox
import cv2
import matplotlib as mpl
import tkinter as tk
from collections import OrderedDict
from dataclasses import dataclass

# WMTD306N100AXM
l_cam_m = np.array([[712.623, 0, 653.448],
                    [0, 712.623, 475.572],
                    [0, 0, 1]])
l_dist_coeff = np.array([0.072867, -0.026286, 0.007135, -0.000997])
# WMTD305L6003D6
r_cam_m = np.array([[716.896, 0, 668.902],
                    [0, 716.896, 460.618],
                    [0, 0, 1]])
r_dist_coeff = np.array([0.07542, -0.026874, 0.006662, -0.000775])

rvec_left = np.array([0.44326239, -1.48492036, -0.37927786])
tvec_left = np.array([-0.07625896, -0.00155683, 0.34419907])
l_blob = np.array([[512.94886659, 491.18943589],
                   [505.12935192, 520.63836956],
                   [468.50093122, 532.21119389],
                   [479.84074949, 574.78916273]])
rvec_right = np.array([1.34109638, -0.88425646, 0.22458526])
tvec_right = np.array([0.03042639, 0.10143083, 0.45047843])
r_blob = np.array([[708.97113752, 661.46590266],
                   [686.86331947, 668.79149335],
                   [666.72478901, 650.31063749],
                   [642.81686566, 667.3940128]])

default_cameraK = np.eye(3).astype(np.float64)
default_distCoeff = np.zeros((4, 1)).astype(np.float64)

# Rotation Matrix to Euler Angles (XYZ)
def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def rotation_matrix_to_quaternion(R):
    qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)

    return np.array([qw, qx, qy, qz])


def calculate_camera_position_direction(rvec, tvec):
    # Extract rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    R_inv = np.linalg.inv(R)
    # Camera position (X, Y, Z)
    Cam_pos = -R_inv @ tvec
    X, Y, Z = Cam_pos.ravel()

    unit_z = np.array([0, 0, 1])

    # idea 1
    # roll = math.atan2(R[2][1], R[2][2])
    # pitch = math.atan2(-R[2][0], math.sqrt(R[2][1]**2 + R[2][2]**2))
    # yaw = math.atan2(R[1][0], R[0][0])

    # idea 2
    Zc = np.reshape(unit_z, (3, 1))
    Zw = np.dot(R_inv, Zc)  # world coordinate of optical axis
    zw = Zw.ravel()

    pan = np.arctan2(zw[1], zw[0]) - np.pi / 2
    tilt = np.arctan2(zw[2], np.sqrt(zw[0] * zw[0] + zw[1] * zw[1]))

    # roll
    unit_x = np.array([1, 0, 0])
    Xc = np.reshape(unit_x, (3, 1))
    Xw = np.dot(R_inv, Xc)  # world coordinate of camera X axis
    xw = Xw.ravel()
    xpan = np.array([np.cos(pan), np.sin(pan), 0])

    roll = np.arccos(np.dot(xw, xpan))  # inner product
    if xw[2] < 0:
        roll = -roll

    roll = roll
    pitch = tilt
    yaw = pan

    print('roll', roll)
    print('pitch', pitch)
    print('yaw', yaw)

    optical_axis = R.T @ unit_z.T
    # 카메라 위치에서 optical axis까지의 방향 벡터 계산
    optical_axis_x, optical_axis_y, optical_axis_z = optical_axis

    return (X, Y, Z), (optical_axis_x, optical_axis_y, optical_axis_z), (roll, pitch, yaw)


def draw_projection(ax, rvec, tvec, cam_m, dist_coeff):
    Rod, _ = cv2.Rodrigues(rvec)
    ret = cv2.projectPoints(points[0:4], Rod, tvec, cam_m, dist_coeff)
    xx, yy = ret[0].reshape(len(points[0:4]), 2).transpose()

    x = [x for x in xx]
    y = [y for y in yy]
    ax.scatter(x, y, marker='o', s=20, color='black', alpha=0.5)

    ax.set_xlim([0, 1280])
    ax.set_ylim([0, 960])

    # y축의 방향을 바꾸어 위에서 아래로 증가하도록 설정
    ax.invert_yaxis()
    # 차트의 가로 세로 비율을 1280:960으로 설정
    ax.set_aspect(1280 / 960)


pos_left, dir_left, _ = calculate_camera_position_direction(rvec_left, tvec_left)
pos_right, dir_right, _ = calculate_camera_position_direction(rvec_right, tvec_right)

print('pos_left', pos_left, 'dir_left', dir_left)
print('pos_right', pos_right, 'dir_right', dir_right)

fig = plt.figure(figsize=(30, 10), num='Camera Simulator')
plt.rcParams.update({'font.size': 7})
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
ax = plt.subplot(gs[0], projection='3d')
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])


# 원점 표시 (scatter plot)
ax.scatter(0, 0, 0, c='k', marker='o', label='Origin')
# LED 위치 표시
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', label='LED Positions')
# 카메라 위치 및 방향 표시 (scatter plot & quiver plot)
ax.scatter(*pos_left, c='r', marker='o', label='Camera Left')
ax.quiver(*pos_left, *dir_left, color='r', label='Direction Left', length=0.1)
ax.scatter(*pos_right, c='g', marker='o', label='Camera Right')
ax.quiver(*pos_right, *dir_right, color='g', label='Direction Right', length=0.1)
ax.set_xlim([-0.5, 0.5])
ax.set_xlabel('X')
ax.set_ylim([-0.5, 0.5])
ax.set_ylabel('Y')
ax.set_zlim([-0.5, 0.5])
ax.set_zlabel('Z')
scale = 1.5
f = zoom_factory(ax, base_scale=scale)

draw_projection(ax1, rvec_left, tvec_left, l_cam_m, l_dist_coeff)
draw_projection(ax2, rvec_right, tvec_right, r_cam_m, r_dist_coeff)

plt.show()
