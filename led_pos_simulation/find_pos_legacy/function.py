import numpy as np
from definition import *


def pickle_data(rw_mode, path, data):
    import pickle
    import gzip
    try:
        if rw_mode == READ:
            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)
            return data
        elif rw_mode == WRITE:
            with gzip.open(path, 'wb') as f:
                pickle.dump(data, f)
        else:
            print('not support mode')
    except:
        print('file r/w error')
        return ERROR


def set_axis_style(ax, alpha):
    for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
        axis.line.set_alpha(alpha)
        axis.pane.set_edgecolor((0.8, 0.8, 0.8, alpha))
        axis.pane.fill = False
        axis.label.set_alpha(alpha)
        axis.gridlines.set_alpha(alpha)

    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks() + ax.zaxis.get_major_ticks():
        tick.label1.set_alpha(alpha)
        tick.tick1line.set_alpha(alpha)
        tick.tick2line.set_alpha(alpha)


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


def check_facing_dot(camera_pos, leds_coords, angle_spec):
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




def sequential_closest_distances(coords):
    visited_indices = [0]
    current_idx = 0
    distances = []

    for _ in range(len(coords) - 1):
        min_dist = float("inf")
        closest_idx = None

        for idx, coord in enumerate(coords):
            if idx not in visited_indices:
                dist = np.linalg.norm(coords[current_idx] - coord) + led_r * 2

                if dist < min_dist:
                    min_dist = dist
                    closest_idx = idx

        visited_indices.append(closest_idx)
        distances.append(min_dist)
        current_idx = closest_idx

    return distances


def draw_sequential_closest_lines(ax, led_coords_o):
    visited_indices = [0]
    current_idx = 0

    for _ in range(len(led_coords_o) - 1):
        min_dist = float("inf")
        closest_idx = None

        for idx, coord in enumerate(led_coords_o):
            if idx not in visited_indices:
                dist = np.linalg.norm(led_coords_o[current_idx] - coord)

                if dist < min_dist:
                    min_dist = dist
                    closest_idx = idx

        visited_indices.append(closest_idx)
        ax.plot(
            [led_coords_o[current_idx][0], led_coords_o[closest_idx][0]],
            [led_coords_o[current_idx][1], led_coords_o[closest_idx][1]],
            [led_coords_o[current_idx][2], led_coords_o[closest_idx][2]],
            linewidth=0.5,
            alpha=0.5,
            color='black',
            linestyle='--'
        )
        current_idx = closest_idx

def select_points(coords, num_leds, min_start_distance=0):
    if num_leds > coords.shape[0]:
        raise ValueError("num_leds cannot be greater than the number of points in coords")

    selected_indices = [0]  # 시작점을 첫 번째로 선택

    for _ in range(num_leds):  # 시작점이 이미 선택되었으므로 num_leds - 1
        min_dists = np.full((coords.shape[0],), np.inf)

        for selected_idx in selected_indices:
            dists = np.linalg.norm(coords - coords[selected_idx], axis=1) + led_r * 2
            min_dists = np.minimum(min_dists, dists)

        # 시작점과의 최소 거리 조건을 적용
        min_dists[:1] = 0  # 시작점 자체의 거리를 0으로 설정
        valid_indices = np.where(min_dists >= min_start_distance)
        
        if valid_indices[0].size == 0:  # 모든 점이 거리 조건을 충족하지 않는 경우
            # raise ValueError("No points found with a distance greater or equal to min_start_distance")
            return ERROR

        next_idx = np.argmax(min_dists[valid_indices])
        selected_indices.append(valid_indices[0][next_idx])

    return coords[selected_indices]



def led_position(*args):
    draw = args[0][0]
    ax = args[0][1]
    radius = args[0][2]
    upper_z = args[0][3]
    lower_z = args[0][4]
    # 구 캡 위의 점들을 계산
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
    if draw == 1:
        ax.scatter(x_masked, y_masked, z_masked, color='gray', marker='.', alpha=0.1)
    coords = np.array([x_masked, y_masked, z_masked]).T
    # 시작점을 기반으로 점 선택
    ret = select_points(coords, num_leds)
    if ret == ERROR:
        return ERROR, ERROR
    led_coords = ret
    padding_mask = ((z >= lower_z - led_r) & (z <= lower_z)) | ((z >= upper_z) & (z <= upper_z + led_r))
    x_masked = x[padding_mask]
    y_masked = y[padding_mask]
    z_masked = z[padding_mask]
    if draw == 1:
        ax.scatter(x_masked, y_masked, z_masked, color='green', marker='.', alpha=0.1)
    padding_coords = np.array([x_masked, y_masked, z_masked]).T

    combined_coords = np.vstack((coords, padding_coords))

    return combined_coords, led_coords

    
def make_camera_position(ax, radius):
    # 구 캡 위의 점들을 계산
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi = np.linspace(0, np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    mask = (z >= C_LOWER_Z) & (z <= C_UPPER_Z)
    x_masked = x[mask]
    y_masked = y[mask]
    z_masked = z[mask]
    ax.scatter(x_masked, y_masked, z_masked, color='gray', marker='.', alpha=0.1)
    coords = np.array([x_masked, y_masked, z_masked]).T

    return coords

def set_plot_option(ax, radius):
    # 플롯 옵션 설정
    ax.set_title(f'Controller {num_leds} LEDs')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # x, y, z 축 범위 설정
    axis_limit = radius * 1.5
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-axis_limit, axis_limit)

    set_axes_equal(ax)
    set_axis_style(ax, 0.5)


def find_led_blobs(*args):
    ax1 = args[0][0]
    ax2 = args[0][1]

    TEMP_R = R
    LOOP_CNT = 0
    UPPER_Z_ANGLE = 0.2
    LOWER_Z_ANGLE = 0.1
    cam_coords = make_camera_position(ax2, CAM_DISTANCE)
    while True:
        cnt = 0
        distance_detect = 0
        upper_z = TEMP_R * UPPER_Z_ANGLE
        lower_z = -TEMP_R * LOWER_Z_ANGLE
        if TEMP_R < 0:
            break
        
        _, ret_data = led_position([0, ax1, TEMP_R, upper_z, lower_z])
        if ret_data == ERROR:
            print('not enough distance from start point')
            TEMP_R += 0.1
            break
        led_coords_o = ret_data[1:]

        if len(led_coords_o) < num_leds:
            print('led_num error')
            break

        distances = sequential_closest_distances(led_coords_o)
        for data in distances:
            temp_distance = data.copy()
            # print(temp_distance)
            if temp_distance < DISTANCE_SPEC:
                print('distance error', temp_distance)
                distance_detect = 1
                break
        if distance_detect == 1:
            print('distance_detect')

        facing_dot_check = 0    
        for camera_pos in cam_coords:        
            facing_pts = check_facing_dot(camera_pos, led_coords_o, ANGLE_SPEC)
            if len(facing_pts) < 4:
                facing_dot_check = 1
                break
            cnt += 1
            
        print('loop', LOOP_CNT, 'R', TEMP_R)

        if cnt == len(cam_coords) and distance_detect == 1:
            print('all position checked')
            break

        if facing_dot_check == 0:
                TEMP_R -= 0.1
        else:
            print('facing dot error')
            break


        LOOP_CNT += 1

    print('TEMP_R', TEMP_R, upper_z, lower_z)

    coords, ret_coords = led_position([1, ax1, TEMP_R, upper_z, lower_z])

    return coords, cam_coords, ret_coords, upper_z, lower_z, TEMP_R