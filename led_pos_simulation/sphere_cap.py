import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

# 구 캡의 반지름 R과 높이 h
R = 11
UPPER_Z = 2.5
LOWER_Z = -1
num_points = 60
num_leds = 12
CAM_DISTANCE = 50
ANGLE_SPEC = 70
BLOB_INFO = {}

camera_matrix = np.array([[712.623, 0., 653.448],
                          [0., 712.623, 475.572],
                          [0., 0., 1.]])

dist_coeff = np.array([[0.072867],
                       [-0.026268],
                       [0.007135],
                       [-0.000997]])

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def spherical_to_cartesian(radius, azimuth, elevation):
    x = radius * np.cos(np.deg2rad(elevation)) * np.cos(np.deg2rad(azimuth))
    y = radius * np.cos(np.deg2rad(elevation)) * np.sin(np.deg2rad(azimuth))
    z = radius * np.sin(np.deg2rad(elevation))
    return np.array([x, y, z])

def camera_position(ax, radius):
    azimuth, elevation = ax.azim, ax.elev
    camera_pos = spherical_to_cartesian(radius, azimuth, elevation)
    return camera_pos

def check_facing_dot(camera_pos, leds_coords, angle_spec=ANGLE_SPEC):
    pts_facing = []
    for i, led_pos in enumerate(leds_coords):
        # print(i, led_pos)
        o_to_led = led_pos - [0, 0, 0]
        led_to_cam = led_pos - camera_pos
        normalize = led_to_cam / np.linalg.norm(led_to_cam)
        facing_dot = np.dot(normalize, o_to_led)
        angle = np.radians(180.0 - angle_spec)
        if facing_dot < np.cos(angle):
            pts_facing.append({
                'idx': i,
                'pos': list(led_pos),
                'dir': list(o_to_led),
                'angle': facing_dot
            })
    return pts_facing

dragging = False

def on_press(event):
    global dragging
    if event.inaxes == ax:
        dragging = True

def on_release(event):
    global dragging
    if event.inaxes == ax:
        dragging = False

def on_scroll(event):
    global ax
    if event.inaxes == ax:
        zoom_factor = 1.1
        if event.button == 'up':
            ax.dist *= 1 / zoom_factor
        elif event.button == 'down':
            ax.dist *= zoom_factor
        ax.dist = np.clip(ax.dist, 1, 100)  # Optionally, set limits for the zoom level
        fig.canvas.draw_idle()


def on_move(event):
    global dragging, camera_quiver, camera_point, origin, led_properties, led_points 
    if event.inaxes == ax and dragging:
        azim, elev = ax.azim, ax.elev
        camera_pos = camera_position(ax, CAM_DISTANCE)
        # print(f"Azimuth: {azim}, Elevation: {elev}, Camera Position: {camera_pos}")
        facing_pts = check_facing_dot(camera_pos, led_coords_o)
        # print(facing_pts)
        # 카메라 위치를 플롯에 동적으로 업데이트합니다.
        camera_quiver.set_segments([np.array([camera_pos, origin])])

        # 이전 카메라 위치를 표시하는 점을 제거하고 새 점을 그립니다.
        camera_point.remove()
        camera_point = ax.scatter(*camera_pos, color='blue', marker='o', s=3)

        # 보이는 LED는 빨간색으로, 보이지 않는 LED는 회색으로 설정합니다.
        facing_indices = [pt['idx'] for pt in facing_pts]
        for i, coord in enumerate(led_coords_o):
            if i in facing_indices:
                led_properties[i]['color'] = 'red'
                led_properties[i]['size'] = 20
                led_properties[i]['alpha'] = 1.0
            else:
                led_properties[i]['color'] = 'lightgray'
                led_properties[i]['size'] = 5
                led_properties[i]['alpha'] = 0.5

        # 기존 LED 점들을 제거하고 새로운 색상으로 다시 그립니다.
        for led_point in led_points:
            led_point.remove()
        led_points = [ax.scatter(coord[0], coord[1], coord[2], color=led_properties[i]['color'], marker='o', s=led_properties[i]['size']) for i, coord in enumerate(led_coords_o)]

        plt.draw()


def select_points(coords, num_leds):
    selected_indices = [0]  # 시작점을 첫 번째로 선택

    for _ in range(num_leds - 1):
        min_dists = np.full((coords.shape[0],), np.inf)

        for selected_idx in selected_indices:
            dists = np.linalg.norm(coords - coords[selected_idx], axis=1)
            min_dists = np.minimum(min_dists, dists)

        next_idx = np.argmax(min_dists)
        selected_indices.append(next_idx)

    return coords[selected_indices]


def led_position(ax, R):
    # 구 캡 위의 점들을 계산
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi = np.linspace(0, np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)

    x = R * np.sin(phi) * np.cos(theta)
    y = R * np.sin(phi) * np.sin(theta)
    z = R * np.cos(phi)

    # z 축 기준으로 -1 이하와 2 이상을 자르기
    mask = (z >= LOWER_Z) & (z <= UPPER_Z)
    x_masked = x[mask]
    y_masked = y[mask]
    z_masked = z[mask]
    ax.scatter(x_masked, y_masked, z_masked, color='gray', marker='.', alpha=0.1)
    coords = np.array([x_masked, y_masked, z_masked]).T
    # 시작점을 기반으로 점 선택
    led_coords = select_points(coords, num_leds)

    return led_coords


# 3D 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 구 캡을 3D 플롯에 추가
led_coords_o = led_position(ax, R)
# led_colors = ['red'] * num_leds
led_properties = [{'color': 'red', 'alpha': 1.0, 'size': 5} for _ in range(num_leds)]

# 전역 변수로 led_points를 선언합니다.
led_points = []

# 초기 LED 점들을 led_points 리스트에 저장합니다.
led_points = [ax.scatter(coord[0], coord[1], coord[2], color='red', marker='o', s=5) for coord in led_coords_o]

# 원점
ax.scatter(0, 0, 0, color='black', marker='o', s=5)

# 시작점
ax.scatter(led_coords_o[0][0], led_coords_o[0][1], led_coords_o[0][2], color=led_properties[0]['color'], alpha=led_properties[0]['alpha'], marker='o', s=led_properties[0]['size'])
ax.text(led_coords_o[0][0], led_coords_o[0][1], led_coords_o[0][2], str(0), color='black', fontsize=5)

for i in range(1, num_leds):
    ax.scatter(led_coords_o[i, 0], led_coords_o[i, 1], led_coords_o[i, 2], color=led_properties[i]['color'], alpha=led_properties[i]['alpha'], marker='o', s=led_properties[i]['size'])
    ax.text(led_coords_o[i, 0], led_coords_o[i, 1], led_coords_o[i, 2], str(i), color='black', fontsize=5)

led_quivers = [ax.quiver(coord[0], coord[1], coord[2], coord[0], coord[1], coord[2], length=5, linewidths=0.2, color='red', normalize=True) for coord in led_coords_o[1:]]


# 플롯 옵션 설정
ax.set_title('Spherical Caps with 24 LEDs')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# x, y, z 축 범위 설정
axis_limit = R * 3
ax.set_xlim(-axis_limit, axis_limit)
ax.set_ylim(-axis_limit, axis_limit)
ax.set_zlim(-axis_limit, axis_limit)

set_axes_equal(ax)

# 카메라 위치를 표시하기 위한 초기 선 객체를 만듭니다.
origin = np.array([0, 0, 0])
camera_pos_init = camera_position(ax, CAM_DISTANCE)
camera_point = ax.scatter(*camera_pos_init, color='blue', marker='o', s=3)

# 카메라 위치를 표시하기 위한 초기 quiver 객체를 만듭니다.
camera_dir_init = camera_pos_init - origin
camera_quiver = ax.quiver(*camera_pos_init, *camera_dir_init, pivot='tail', color='blue', linestyle='--', lw=0.5, alpha=0.5, arrow_length_ratio=0.1)
camera_quiver.set_segments([np.array([camera_pos_init, origin])])

# 마우스 이벤트 처리를 위한 콜백 함수 연결
cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
cid_release = fig.canvas.mpl_connect('button_release_event', on_release)
cid_move = fig.canvas.mpl_connect('motion_notify_event', on_move)
cid_scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)

plt.show()
