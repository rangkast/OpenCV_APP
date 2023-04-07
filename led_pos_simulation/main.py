import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import TextBox
import cv2
from function import *
import matplotlib as mpl
import tkinter as tk

root = tk.Tk()
width_px = root.winfo_screenwidth()
height_px = root.winfo_screenheight()

# 모니터 해상도에 맞게 조절
mpl.rcParams['figure.dpi'] = 120  # DPI 설정
monitor_width_inches = width_px / mpl.rcParams['figure.dpi']  # 모니터 너비를 인치 단위로 변환
monitor_height_inches = height_px / mpl.rcParams['figure.dpi']  # 모니터 높이를 인치 단위로 변환

# 구 캡의 반지름 R과 높이 h
# 단위 cm
R = 11
r = 0.3
UPPER_Z = 2.5
LOWER_Z = -1
num_points = 150
num_leds = 15
CAM_DISTANCE = 30
ANGLE_SPEC = 70

env_info = f" R:{R} r:{r}\n UPPER_Z:{UPPER_Z} LOWER_Z:{LOWER_Z}\n" \
           f" num_points:{num_points} num_leds:{num_leds}" \
           f" CAM_DISTANCE:{CAM_DISTANCE} ANGLE_SPEC:{ANGLE_SPEC}"

camera_matrix = np.array([[712.623, 0., 653.448],
                          [0., 712.623, 475.572],
                          [0., 0., 1.]])

dist_coeff = np.array([[0.072867],
                       [-0.026268],
                       [0.007135],
                       [-0.000997]])

dragging = False


def on_press(event):
    global dragging
    if event.inaxes == ax1:
        dragging = True


def on_release(event):
    global dragging
    if event.inaxes == ax1:
        dragging = False


def on_scroll(event):
    global ax1
    if event.inaxes == ax1:
        zoom_factor = 1.1
        if event.button == 'up':
            ax1.dist *= 1 / zoom_factor
        elif event.button == 'down':
            ax1.dist *= zoom_factor
        ax1.dist = np.clip(ax1.dist, 1, 100)  # Optionally, set limits for the zoom level
        fig.canvas.draw_idle()


def on_move(event):
    global dragging, camera_quiver, camera_point, origin, led_properties, led_points, info_box
    if event.inaxes == ax1 and dragging:
        azim, elev = ax1.azim, ax1.elev
        camera_pos = camera_position(ax1, CAM_DISTANCE)
        # print(f"Azimuth: {azim}, Elevation: {elev}, Camera Position: {camera_pos}")
        facing_pts = check_facing_dot(camera_pos, led_coords_o, ANGLE_SPEC)
        # print(facing_pts)
        # 카메라 위치를 플롯에 동적으로 업데이트합니다.
        camera_quiver.set_segments([np.array([camera_pos, origin])])

        # 이전 카메라 위치를 표시하는 점을 제거하고 새 점을 그립니다.
        camera_point.remove()
        camera_point = ax1.scatter(*camera_pos, color='blue', marker='o', s=3)

        # 보이는 LED는 빨간색으로, 보이지 않는 LED는 회색으로 설정합니다.
        facing_indices = [pt['idx'] for pt in facing_pts]
        for i, coord in enumerate(led_coords_o):
            if i in facing_indices:
                led_properties[i]['color'] = 'red'
                led_properties[i]['size'] = 25
                led_properties[i]['alpha'] = 1.0
            else:
                led_properties[i]['color'] = 'lightgray'
                led_properties[i]['size'] = 5
                led_properties[i]['alpha'] = 0.5

        # 기존 LED 점들을 제거하고 새로운 색상으로 다시 그립니다.
        for led_point in led_points:
            led_point.remove()
        led_points = [ax1.scatter(coord[0], coord[1], coord[2], color=led_properties[i]['color'], marker='o',
                                  s=led_properties[i]['size']) for i, coord in enumerate(led_coords_o)]

        # 텍스트 박스에 정보 업데이트
        info_text = f" ENVIRONMENT:\n{env_info}\n\n" \
                    f" LOG:\n" \
                    f" Camera Position: {camera_pos}\n Azimuth: {azim}\n Elevation: {elev}\n Facing LEDs: {facing_indices}" \
                    f"\n Count: {len(facing_indices)}"
        info_box.set_val(info_text)
        plt.draw()


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
    led_coords = select_points(coords, num_leds, r)

    return led_coords


# Figure 크기를 모니터 해상도에 맞게 조절하고 제목 추가
fig = plt.figure(figsize=(monitor_width_inches, monitor_height_inches), num='LED Position FinDer')
plt.rcParams.update({'font.size': 7})
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 2])
ax1 = plt.subplot(gs[0], projection='3d')
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax2.axis('off')

# 텍스트 박스 생성
info_box_ax = ax2
info_box = TextBox(info_box_ax, '', initial="")

# 구 캡을 3D 플롯에 추가
ret_coords = led_position(ax1, R)
led_coords_o = ret_coords[1:]
# led_colors = ['red'] * num_leds
led_properties = [{'color': 'red', 'alpha': 1.0, 'size': 5} for _ in range(num_leds)]

# 전역 변수로 led_points를 선언
led_points = []

# 초기 LED 점들을 led_points 리스트에 저장
led_points = [ax1.scatter(coord[0], coord[1], coord[2], color='red', marker='o', s=5) for coord in led_coords_o]
# 원점
ax1.scatter(0, 0, 0, color='black', marker='o', s=5)

# 시작점
ax1.scatter(ret_coords[0][0], ret_coords[0][1], ret_coords[0][2], color='black',
            alpha=led_properties[0]['alpha'], marker='o', s=led_properties[0]['size'])
ax1.text(ret_coords[0][0], ret_coords[0][1], ret_coords[0][2], 'S', color='black', fontsize=10)

# 그 외 점
for i in range(0, num_leds):
    ax1.scatter(led_coords_o[i, 0], led_coords_o[i, 1], led_coords_o[i, 2], color=led_properties[i]['color'],
                alpha=led_properties[i]['alpha'], marker='o', s=led_properties[i]['size'])
    ax1.text(led_coords_o[i, 0], led_coords_o[i, 1], led_coords_o[i, 2], str(i), color='black', fontsize=7)

led_quivers = [
    ax1.quiver(coord[0], coord[1], coord[2], coord[0], coord[1], coord[2], length=5, linewidths=0.2, color='red',
               normalize=True) for coord in led_coords_o[0:]]

# 플롯 옵션 설정
ax1.set_title(f'Controller {num_leds} LEDs')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# x, y, z 축 범위 설정
axis_limit = R * 1.5
ax1.set_xlim(-axis_limit, axis_limit)
ax1.set_ylim(-axis_limit, axis_limit)
ax1.set_zlim(-axis_limit, axis_limit)

set_axes_equal(ax1)
set_axis_style(ax1, 0.5)

# LED의 가장 가까운 거리를 계산하고 stem plot을 그립니다.
distances = sequential_closest_distances(ret_coords)
# 선을 그리기 위한 함수 호출
draw_sequential_closest_lines(ax1, ret_coords)

markerline, stemline, baseline = ax3.stem(range(num_leds), distances, use_line_collection=True)
plt.setp(markerline, markersize=5, alpha=0.5)
plt.setp(stemline, linewidth=0.5, alpha=0.5)
plt.setp(baseline, visible=False)
# stem plot 끝에 값 표시
for i, (xi, yi) in enumerate(zip(range(num_leds), distances)):
    ax3.annotate(f'{yi:.4f}', (xi, yi), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

ax3.set_ylabel("Distance (cm)")
ax3.set_title("Closest Distance for each LED")

# y축 눈금 값을 소수점 두 번째 자리로 설정합니다.
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# 카메라 위치를 표시하기 위한 초기 선 객체를 만듭니다.
origin = np.array([0, 0, 0])
camera_pos_init = camera_position(ax1, CAM_DISTANCE)
camera_point = ax1.scatter(*camera_pos_init, color='blue', marker='o', s=3)

# 카메라 위치를 표시하기 위한 초기 quiver 객체를 만듭니다.
camera_dir_init = camera_pos_init - origin
camera_quiver = ax1.quiver(*camera_pos_init, *camera_dir_init, pivot='tail', color='blue', linestyle='--', lw=0.5,
                           alpha=0.5, arrow_length_ratio=0.1)
camera_quiver.set_segments([np.array([camera_pos_init, origin])])

# 마우스 이벤트 처리를 위한 콜백 함수 연결
cid_press = fig.canvas.mpl_connect('button_press_event', on_press)
cid_release = fig.canvas.mpl_connect('button_release_event', on_release)
cid_move = fig.canvas.mpl_connect('motion_notify_event', on_move)
cid_scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)

# plt.tight_layout()
plt.show()
