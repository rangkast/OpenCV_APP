import numpy as np


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


def select_points(coords, num_leds, r):
    selected_indices = [0]  # 시작점을 첫 번째로 선택

    for _ in range(num_leds):
        min_dists = np.full((coords.shape[0],), np.inf)

        for selected_idx in selected_indices:
            dists = np.linalg.norm(coords - coords[selected_idx], axis=1) + r * 2
            min_dists = np.minimum(min_dists, dists)

        next_idx = np.argmax(min_dists)
        selected_indices.append(next_idx)

    return coords[selected_indices]


def sequential_closest_distances(coords):
    visited_indices = [0]
    current_idx = 0
    distances = []

    for _ in range(len(coords) - 1):
        min_dist = float("inf")
        closest_idx = None

        for idx, coord in enumerate(coords):
            if idx not in visited_indices:
                dist = np.linalg.norm(coords[current_idx] - coord)

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
