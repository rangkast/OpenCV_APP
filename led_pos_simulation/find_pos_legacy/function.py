from definition import *

# rift s data
points = np.array([
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


def zoom_factory(ax, base_scale=2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_zlim = ax.get_zlim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
        cur_zrange = (cur_zlim[1] - cur_zlim[0]) * .5
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
        ax.set_zlim([(xdata + ydata) / 2 - cur_zrange * scale_factor,
                     (xdata + ydata) / 2 + cur_zrange * scale_factor])
        # force re-draw
        plt.draw()

    # get the figure of interest
    fig = ax.get_figure()
    # attach the call back
    fig.canvas.mpl_connect('scroll_event', zoom_fun)

    # return the function
    return zoom_fun


class Vector:
    def __init__(self, *coords):
        if len(coords) == 1 and isinstance(coords[0], (list, tuple, np.ndarray)):
            coords = coords[0]
        self.coords = np.array(coords)
        self.x = coords[0] if len(coords) > 0 else 0
        self.y = coords[1] if len(coords) > 1 else 0
        self.z = coords[2] if len(coords) > 2 else 0

    def __repr__(self):
        return f"Vector({', '.join(map(str, self.coords))})"

    def __getitem__(self, index):
        return self.coords[index]

    def normalized(self):
        norm = np.linalg.norm(self.coords)
        return Vector(*(self.coords / norm))


def pickle_data(rw_mode, path, data):
    import pickle
    import gzip
    print('path', path)
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


def convert_to_meters(data):
    for key in data:
        data[key] = [(coord[0] * 0.01, coord[1] * 0.01, coord[2] * 0.01) for coord in data[key]]
    return data


trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


class POSE_ESTIMATION_METHOD(Enum):
    # Opencv legacy
    SOLVE_PNP_RANSAC = auto()
    SOLVE_PNP_REFINE_LM = auto()
    SOLVE_PNP_AP3P = auto()
    SOLVE_PNP = auto()
    SOLVE_PNP_RESERVED = auto()


# Default solvePnPRansac
def solvepnp_ransac(*args):
    cam_id = args[0][0]
    points3D = args[0][1]
    points2D = args[0][2]
    camera_k = args[0][3]
    dist_coeff = args[0][4]
    # check assertion
    if len(points3D) != len(points2D):
        print("assertion len is not equal")
        return ERROR, NOT_SET, NOT_SET, NOT_SET

    if len(points2D) < 4:
        print("assertion < 4: ")
        return ERROR, NOT_SET, NOT_SET, NOT_SET

    ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(points3D, points2D,
                                                    camera_k,
                                                    dist_coeff)

    return SUCCESS if ret == True else ERROR, rvecs, tvecs, inliers


# solvePnPRansac + RefineLM
def solvepnp_ransac_refineLM(*args):
    cam_id = args[0][0]
    points3D = args[0][1]
    points2D = args[0][2]
    camera_k = args[0][3]
    dist_coeff = args[0][4]
    ret, rvecs, tvecs, inliers = solvepnp_ransac(points3D, points2D, camera_k, dist_coeff)
    # Do refineLM with inliers
    if ret == SUCCESS:
        if not hasattr(cv2, 'solvePnPRefineLM'):
            print('solvePnPRefineLM requires OpenCV >= 4.1.1, skipping refinement')
        else:
            assert len(inliers) >= 3, 'LM refinement requires at least 3 inlier points'
            # refine r_t vector and maybe changed
            cv2.solvePnPRefineLM(points3D[inliers],
                                 points2D[inliers], camera_k, dist_coeff,
                                 rvecs, tvecs)

    return SUCCESS if ret == True else ERROR, rvecs, tvecs, NOT_SET


# solvePnP_AP3P, 3 or 4 points need
def solvepnp_AP3P(*args):
    cam_id = args[0][0]
    points3D = args[0][1]
    points2D = args[0][2]
    camera_k = args[0][3]
    dist_coeff = args[0][4]

    # check assertion
    if len(points3D) != len(points2D):
        print("assertion len is not equal")
        return ERROR, NOT_SET, NOT_SET, NOT_SET

    if len(points2D) < 3 or len(points2D) > 4:
        print("assertion ", len(points2D))
        return ERROR, NOT_SET, NOT_SET, NOT_SET

    ret, rvecs, tvecs = cv2.solvePnP(points3D, points2D,
                                     camera_k,
                                     dist_coeff,
                                     flags=cv2.SOLVEPNP_AP3P)

    return SUCCESS if ret == True else ERROR, rvecs, tvecs, NOT_SET


SOLVE_PNP_FUNCTION = {
    POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC: solvepnp_ransac,
    POSE_ESTIMATION_METHOD.SOLVE_PNP_REFINE_LM: solvepnp_ransac_refineLM,
    POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P: solvepnp_AP3P,
}


def view_camera_infos(frame, text, x, y):
    cv2.putText(frame, text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), lineType=cv2.LINE_AA)


def rw_json_data(rw_mode, path, data):
    try:
        if rw_mode == READ:
            with open(path, 'r', encoding="utf-8") as rdata:
                json_data = json.load(rdata)
            return json_data
        elif rw_mode == WRITE:
            with open(path, 'w', encoding="utf-8") as wdata:
                json.dump(data, wdata, ensure_ascii=False, indent="\t")
        else:
            print('not support mode')
    except:
        # print('file r/w error')
        return ERROR


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.legacy.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


def find_center(frame, SPEC_AREA):
    x_sum = 0
    t_sum = 0
    y_sum = 0
    g_c_x = 0
    g_c_y = 0
    m_count = 0

    (X, Y, W, H) = SPEC_AREA

    for y in range(Y, Y + H):
        for x in range(X, X + W):
            x_sum += x * frame[y][x]
            t_sum += frame[y][x]
            m_count += 1

    for x in range(X, X + W):
        for y in range(Y, Y + H):
            y_sum += y * frame[y][x]

    if t_sum != 0:
        g_c_x = x_sum / t_sum
        g_c_y = y_sum / t_sum

    if g_c_x == 0 or g_c_y == 0:
        return 0, 0
    #

    result_data_str = f'{g_c_x}' + f'{g_c_y}'
    print(result_data_str)

    return g_c_x, g_c_y


def load_data(path):
    image_files = glob.glob(os.path.join(path, "*.png"))
    data_files = glob.glob(os.path.join(path, "*.txt"))
    images = [cv2.imread(img) for img in image_files]
    return images, image_files, data_files


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


def remake_3d_point(camera_k_0, camera_k_1, RT_0, RT_1, BLOB_0, BLOB_1):
    l_rotation, _ = cv2.Rodrigues(RT_0['rvec'])
    r_rotation, _ = cv2.Rodrigues(RT_1['rvec'])
    l_projection = np.dot(camera_k_0, np.hstack((l_rotation, RT_0['tvec'])))
    r_projection = np.dot(camera_k_1, np.hstack((r_rotation, RT_1['tvec'])))
    l_blob = np.reshape(BLOB_0, (1, len(BLOB_0), 2))
    r_blob = np.reshape(BLOB_1, (1, len(BLOB_1), 2))
    triangulation = cv2.triangulatePoints(l_projection, r_projection,
                                          l_blob, r_blob)
    homog_points = triangulation.transpose()
    get_points = cv2.convertPointsFromHomogeneous(homog_points)

    return get_points