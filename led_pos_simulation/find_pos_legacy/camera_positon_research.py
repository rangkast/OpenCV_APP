from vector_data import *

def zoom_factory(ax, base_scale=2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
        xdata = event.xdata
        ydata = event.ydata
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange * scale_factor,
                     xdata + cur_xrange * scale_factor])
        ax.set_ylim([ydata - cur_yrange * scale_factor,
                     ydata + cur_yrange * scale_factor])
        # force re-draw
        plt.draw()

    # get the figure of interest
    fig = ax.get_figure()
    # attach the call back
    fig.canvas.mpl_connect('scroll_event', zoom_fun)

    # return the function
    return zoom_fun


led_positions = np.array([
    [-0.02146761, -0.00343424, -0.01381839],
    [-0.0318701, 0.00568587, -0.01206734],
    [-0.03692925, 0.00930785, 0.00321071],
    [-0.04287211, 0.02691347, -0.00194137],
    [-0.04170018, 0.03609551, 0.01989264],
    [-0.02923584, 0.06186962, 0.0161972],
    [-0.01456789, 0.06295633, 0.03659283],
    [0.00766914, 0.07115411, 0.0206431],
    [0.02992447, 0.05507271, 0.03108736],
    [0.03724313, 0.05268665, 0.01100446],
    [0.04265723, 0.03016438, 0.01624689],
    [0.04222733, 0.0228845, -0.00394005],
    [0.03300807, 0.00371497, 0.00026865],
    [0.03006234, 0.00378822, -0.01297127]
])

# 이미 계산된 카메라 매개변수
camera_calibration = {"serial": "WMTD303A5006BW", "camera_f": [714.938, 714.938], "camera_c": [676.234, 495.192], "camera_k": [0.074468, -0.024896, 0.005643, -0.000568]}
A = np.array([[camera_calibration['camera_f'][0], 0, camera_calibration['camera_c'][0]],
              [0, camera_calibration['camera_f'][1], camera_calibration['camera_c'][1]],
              [0, 0, 1]])
distCoeffs = camera_calibration['camera_k']

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


default_cameraK = np.eye(3).astype(np.float64)

print(default_cameraK)
# left
rvec_left = np.array([0.436, -1.345, -0.404])
tvec_left = np.array([-0.037, 0.004, 0.334])

# right
rvec_right = np.array([1.422, -0.829, 0.242])
tvec_right = np.array([0.054, 0.102, 0.435])

pos_left, dir_left, _ = calculate_camera_position_direction(rvec_left, tvec_left)
pos_right, dir_right, _ = calculate_camera_position_direction(rvec_right, tvec_right)

print('pos_left', pos_left, 'dir_left', dir_left)
print('pos_right', pos_right, 'dir_right', dir_right)

# 3D plot 생성
fig = plt.figure(figsize=(10, 10))  # 플롯 크기 조절
ax = fig.add_subplot(111, projection='3d')

# 원점 표시 (scatter plot)
ax.scatter(0, 0, 0, c='k', marker='o', label='Origin')

# LED 위치 표시
ax.scatter(led_positions[:, 0], led_positions[:, 1], led_positions[:, 2], c='b', marker='o', label='LED Positions')

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

plt.legend()
plt.show()