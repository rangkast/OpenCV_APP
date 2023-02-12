from robot_system.resource.definition import *
from robot_system.resource.robot_system_data import *
from robot_system.resource.vectors_data import *


warnings.simplefilter(action='ignore', category=FutureWarning)
TAG = '[COMMON]'


def terminal_cmd(cmd_m, cmd_s):
    print(TAG, terminal_cmd.__name__)
    try:
        result = subprocess.run([cmd_m, cmd_s], stdout=subprocess.PIPE).stdout.decode('utf-8')
        device_re = re.compile(b"Bus\s+(?P<bus>\d+)\s+Device\s+(?P<device>\d+).+ID\s(?P<id>\w+:\w+)\s(?P<tag>.+)$",
                               re.I)
        df = subprocess.check_output("lsusb")
        devices = []
        for i in df.split(b'\n'):
            if i:
                info = device_re.match(i)
                if info:
                    dinfo = info.groupdict()
                    dinfo['device'] = '/dev/bus/usb/%s/%s' % (dinfo.pop('bus'), dinfo.pop('device'))
                    devices.append(dinfo)
    except:
        print(TAG, 'exception')
        traceback.print_exc()
    else:
        print(TAG, 'done')
    finally:
        if DEBUG > DEBUG_LEVEL.LV_2:
            print(TAG, ' -> ', devices)
    temp = result.split('\n\n')
    print(TAG, "==================================================")
    ret_val = []
    for i in range(len(temp)):
        if ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_system_name() in temp[i]:
            ret_val.append(temp[i])
            print("add camera dev", temp[i])
        else:
            print("skipping camera", temp[i])
    print(TAG, "==================================================")
    return ret_val


def init_data_array(cam_dev_list):
    print(TAG, init_data_array.__name__)
    print(cam_dev_list)
    camera_info_array = []

    if ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_system_name() == SENSOR_NAME_RIFT:
        for i in range(len(cam_dev_list)):
            cam_json = ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_camera_json()
            print(TAG, 'cam_id[', i, '] :', cam_json[i])
            jsonObject = json.load(
                open(''.join(['jsons/cam_info/', f'{SENSOR_NAME_RIFT}/calibration/', f'{cam_json[i]}'])))
            video_dev = cam_dev_list[i].split('\n\t')
            '''                        
              k = [ k₁ k₂, k₃, k4 ] for CV1 fisheye distortion                    

                  ⎡ fx 0  cx ⎤
              A = ⎢ 0  fy cy ⎥
                  ⎣ 0  0  1  ⎦          
            '''
            f = jsonObject.get('camera_f')
            c = jsonObject.get('camera_c')
            k = jsonObject.get('camera_k')

            A = np.array([[f[0], 0.0, c[0]],
                          [0.0, f[1], c[1]],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
            K = np.array([[k[0]], [k[1]], [k[2]], [k[3]]], dtype=np.float64)

            if DEBUG > DEBUG_LEVEL.DISABLE:
                print(TAG, 'cameraK: ', A)
                print(TAG, 'dist_coeff: ', K)

            temp_data = CAMERA_INFO_DATA(SENSOR_NAME_RIFT, jsonObject.get('serial'), i, video_dev[1],
                                         DISPLAY_DATA(CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, DEGREE_0),
                                         CAMERA_CALIBRATION_DATA(A, K))
            temp_data.print_all()
            camera_info_array.append(temp_data)

    elif ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_system_name() == SENSOR_NAME_DROID:
        for i in range(len(cam_dev_list)):
            video_dev = cam_dev_list[i].split('\n\t')
            for idx, dev_name in enumerate(video_dev):
                if 'dev' in dev_name:
                    if DEBUG > DEBUG_LEVEL.DISABLE:
                        print(TAG, dev_name)
                    cam_json = ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_camera_json()
                    jsonObject = json.load(
                        open(''.join(['jsons/cam_info/', f'{SENSOR_NAME_DROID}/', f'{cam_json[idx - 1]}'])))
                    temp_data = CAMERA_INFO_DATA(SENSOR_NAME_DROID, jsonObject.get('serial'), idx - 1, dev_name,
                                                 DISPLAY_DATA(CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT,
                                                              int(jsonObject.get('rotate'))),
                                                 CAMERA_CALIBRATION_DATA(
                                                     np.array(jsonObject.get('camera_k'), dtype=np.float64),
                                                     np.array(jsonObject.get('distcoeff'), dtype=np.float64)))
                    temp_data.print_all()
                    camera_info_array.append(temp_data)

    for camera_info_result in camera_info_array:
        print(TAG, camera_info_result)

    return camera_info_array


def init_coord_json(file):
    print(TAG, init_coord_json.__name__)
    try:
        json_file = open(''.join(['jsons/specs/', f'{file}', '.json']))
        jsonObject = json.load(json_file)
        model_points = jsonObject.get('TrackedObject').get('ModelPoints')
        pts = [0 for i in range(len(model_points))]
        for data in model_points:
            idx = data.split('Point')[1]
            x = model_points.get(data)[0]
            y = model_points.get(data)[1]
            z = model_points.get(data)[2]
            u = model_points.get(data)[3]
            v = model_points.get(data)[4]
            w = model_points.get(data)[5]
            r1 = model_points.get(data)[6]
            r2 = model_points.get(data)[7]
            r3 = model_points.get(data)[8]

            pts[int(idx)] = LED_DATA(idx, [x, y, z], [u, v, w], [r1, r2, r3])

            print(TAG, ''.join(['{ .pos = {{', f'{x}', ',', f'{y}', ',', f'{z}',
                                ' }}, .dir={{', f'{u}', ',', f'{v}', ',', f'{w}', ' }}, .pattern=', f'{idx}', '},']))
    except:
        print(TAG, 'exception')
        traceback.print_exc()
    finally:
        ROBOT_SYSTEM_DATA[SYSTEM_SETTING].set_led_cnt(len(model_points))
        print(TAG, 'done')
    return pts


def rotate(src, cam_id):
    _, _, degrees = ROBOT_SYSTEM_DATA[CAM_INFO][cam_id].get_display_info()

    if degrees == DEGREE_90:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 1)

    elif degrees == DEGREE_180:
        dst = cv2.flip(src, -1)

    elif degrees == DEGREE_270:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 0)
    else:
        dst = src
    return dst


def set_display_setting(src, cam_id):
    W, H, degrees = ROBOT_SYSTEM_DATA[CAM_INFO][cam_id].get_display_info()

    ROBOT_SYSTEM_DATA[CAM_INFO][cam_id].set_display_info(CAP_PROP_FRAME_WIDTH,
                                                         CAP_PROP_FRAME_HEIGHT,
                                                         degrees)

    if ROBOT_SYSTEM_DATA[CAM_INFO][cam_id].get_dev() == SENSOR_NAME_RIFT:
        src.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        src.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    if DEBUG > DEBUG_LEVEL.DISABLE:
        print(TAG, 'cam_id: ', cam_id, ' display size: %d, %d' % (W, H))


def view_camera_infos(frame, text, x, y):
    cv2.putText(frame, text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), lineType=cv2.LINE_AA)


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
            print(TAG, 'not support mode')
    except:
        print(TAG, 'file r/w error')
        return ERROR


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
            print(TAG, 'not support mode')
    except:
        print(TAG, 'file r/w error')
        return ERROR


def load_string_yaml(file_name):
    try:
        with open(file_name) as f:
            file = yaml.load(f, Loader=yaml.FullLoader)
            return file
    except:
        print(TAG, 'file r/w error')
        return ERROR


def draw_common_ui(draw_frame, main_str, cam_l_id, cam_r_id, cam_cnt):
    try:
        W, H, _ = ROBOT_SYSTEM_DATA[CAM_INFO][cam_l_id].get_display_info()
        view_camera_infos(draw_frame, f'{main_str}', W * 2 - 250, H - 35)

        # Draw LEFT Camera Info
        imgL_name = ROBOT_SYSTEM_DATA[CAM_INFO][cam_l_id].get_name()
        view_camera_infos(draw_frame, 'cam [' + f'{cam_cnt}' + f'] {imgL_name}', 30, 35)
        cv2.rectangle(draw_frame, (5, 5), (W - 5, H - 5),
                      (255, 255, 255), 2)
        cv2.circle(draw_frame, (int(W / 2), int(H / 2)), 2, color=(0, 0, 255),
                   thickness=-1)

        # Draw Right Camera Info
        imgR_name = ROBOT_SYSTEM_DATA[CAM_INFO][cam_r_id].get_name()
        view_camera_infos(draw_frame, 'cam [' + f'{cam_cnt + 1}' + f'] {imgR_name}', W + 30, 35)
        cv2.rectangle(draw_frame, (W + 5, 5), (W * 2 - 5, H - 5),
                      (255, 255, 255), 2)
        cv2.circle(draw_frame, (int(W / 2) + W, int(H / 2)), 2, color=(0, 0, 255),
                   thickness=-1)
        lr_position = L_CONTROLLER if ROBOT_SYSTEM_DATA[
                                          SYSTEM_SETTING].get_lr_position() == LR_POSITION.LEFT else R_CONTROLLER
        view_camera_infos(draw_frame, f'{lr_position}', W * 2 - 250, H - 70)

        if ROBOT_SYSTEM_DATA[SYSTEM_SETTING].check_functions('robot_animate_tracker') != SUCCESS:
            if main_str == SYSTEM_SETTING_MODE:
                l_r_t = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_cnt].get_r_t().get_curr_r_t()
                if len(l_r_t.rvecs) > 0:
                    l_rvecs = np.round_(get_euler_from_quat('xyz', R.from_rotvec(l_r_t.rvecs.reshape(3)).as_quat()), 3)
                    l_tvecs = l_r_t.tvecs.reshape(3)
                    cv2.putText(draw_frame, ''.join(['rot 'f'{l_rvecs}']),
                                (W - 400, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

                    cv2.putText(draw_frame, ''.join(['pos 'f'{l_tvecs}']),
                                (W - 400, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

                r_r_t = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_cnt + 1].get_r_t().get_curr_r_t()
                if len(r_r_t.rvecs) > 0:
                    r_rvecs = np.round_(get_euler_from_quat('xyz', R.from_rotvec(r_r_t.rvecs.reshape(3)).as_quat()), 3)
                    r_tvecs = r_r_t.tvecs.reshape(3)
                    cv2.putText(draw_frame, ''.join(['rot 'f'{r_rvecs}']),
                                (W * 2 - 400, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

                    cv2.putText(draw_frame, ''.join(['pos 'f'{r_tvecs}']),
                                (W * 2 - 400, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)

        if main_str == SYSTEM_SETTING_MODE or main_str == CALIBRATION_MODE:
            count = int((len(ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_cnt].get_blob().get_acc_blobs()) +
                         len(ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_cnt + 1].get_blob().get_acc_blobs())) / 2)
            view_camera_infos(draw_frame, f'{count}', W * 2 - 250, H - 100)

    except:
        traceback.print_exc()


def get_spec_data_from_json(json_data, imgL_name, imgR_name, cam_l_id, cam_r_id):
    temp_bboxes = []

    if json_data != ERROR and json_data is not None:
        try:
            lr_position = L_CONTROLLER if ROBOT_SYSTEM_DATA[
                                              SYSTEM_SETTING].get_lr_position() == LR_POSITION.LEFT else R_CONTROLLER
            if str(cam_l_id) in json_data:
                for i, data_i in enumerate(json_data[str(cam_l_id)]):
                    if imgL_name in data_i:
                        if lr_position in json_data[str(cam_l_id)][i][imgL_name]:
                            for left_info in json_data[str(cam_l_id)][i][imgL_name][lr_position]['spec']:
                                temp_bboxes.append(
                                    {'idx': left_info['idx'], 'bbox': left_info['bbox'], 'side': left_info['side']})
            if str(cam_r_id) in json_data:
                for i, data_i in enumerate(json_data[str(cam_r_id)]):
                    if imgR_name in data_i:
                        if lr_position in json_data[str(cam_r_id)][i][imgR_name]:
                            for right_info in json_data[str(cam_r_id)][i][imgR_name][lr_position]['spec']:
                                temp_bboxes.append(
                                    {'idx': right_info['idx'], 'bbox': right_info['bbox'], 'side': right_info['side']})
        except:
            traceback.print_exc()
            print(TAG, 'ERROR')
            return []

    return temp_bboxes


def draw_dots(ax, *args):
    flag = args[0][0]
    c = args[0][1]
    if flag == D3D:
        pts = ROBOT_SYSTEM_DATA[LED_INFO]
        x = [led_data.get_pos()[0] for led_data in pts]
        y = [led_data.get_pos()[1] for led_data in pts]
        z = [led_data.get_pos()[2] for led_data in pts]
        u = [led_data.get_dir()[0] for led_data in pts]
        v = [led_data.get_dir()[1] for led_data in pts]
        w = [led_data.get_dir()[2] for led_data in pts]
        idx = [led_data.get_idx() for led_data in pts]
        ax.scatter(x, y, z, marker='o', s=3, color=c, alpha=0.5)
        ax.quiver(x, y, z, u, v, w, length=0.05, linewidths=0.1, color='red', normalize=True)
        for idx, x, y, z in zip(idx, x, y, z):
            label = '%s' % idx
            ax.text(x, y, z, label, size=5)
    elif flag == D3DR:
        for key, pts in ROBOT_SYSTEM_DATA[GROUP_DATA_INFO].items():
            group_num = int(key)
            for i in range(len(pts)):
                if len(pts[i].get_pair_xy()) > 0:
                    x = [pts[i].get_remake_3d().blob.x]
                    y = [pts[i].get_remake_3d().blob.y]
                    z = [pts[i].get_remake_3d().blob.z]
                    idx = [pts[i].get_remake_3d().blob.idx]
                    ax.scatter(x, y, z, marker='o', s=7, color=c, alpha=0.5)
                    for idx, x, y, z in zip(idx, x, y, z):
                        label = '%s' % idx
                        ax.text(x, y, z, label, size=5)

    else:
        print(TAG, 'ERROR Not support ', flag)
        return ERROR

    return SUCCESS


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


def show_3d_plot():
    plt.style.use('default')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # draw origin data
    draw_dots(ax, [D3D, 'blue'])
    # draw remake 3d
    draw_dots(ax, [D3DR, 'red'])

    ax.scatter(0, 0, 0, marker='o', color='k', s=20)
    ax.set_xlim([-0.2, 0.2])
    ax.set_xlabel('X')
    ax.set_ylim([-0.2, 0.2])
    ax.set_ylabel('Y')
    ax.set_zlim([-0.2, 0.2])
    ax.set_zlabel('Z')
    scale = 1.5
    f = zoom_factory(ax, base_scale=scale)


def show_stem_plot():
    led_distance = []
    led_num = []
    pts = ROBOT_SYSTEM_DATA[LED_INFO]
    for i in range(len(pts)):
        if len(pts[i].get_remake_3d()) > 0:
            for ii in range(len(pts[i].get_remake_3d())):
                diff_x = '%0.12f' % (
                        pts[i].get_pos()[0] - pts[i].get_remake_3d()[ii].blob.x)
                diff_y = '%0.12f' % (
                        pts[i].get_pos()[1] - pts[i].get_remake_3d()[ii].blob.y)
                diff_z = '%0.12f' % (
                        pts[i].get_pos()[2] - pts[i].get_remake_3d()[ii].blob.z)
                led_num.append(f'LED {i}')
                led_distance.append(np.sqrt(
                    np.power(float(diff_x), 2) + np.power(float(diff_y), 2) + np.power(
                        float(diff_z), 2)))

    if len(led_distance) > 0:
        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        plt.style.use('default')
        plt.figure(figsize=(15, 10))

        markers, stemlines, baseline = plt.stem(led_num, led_distance, label='remake')
        markers.set_color('black')
        plt.legend()


def make_measurement_data(*args):
    ROBOT_SYSTEM_DATA[MEASUREMENT_INFO].append(MEASUREMENT_DATA())
    return SUCCESS


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(TAG, f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(TAG, f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    if DEBUG > DEBUG_LEVEL.LV_2:
        if np.linalg.matrix_rank(H) < 3:
            raise ValueError(TAG, "rank of H = {}, expecting 3".format(np.linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print(TAG, "det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def DLT(P1, P2, point1, point2):
    A = [point1[1] * P1[2, :] - P1[1, :],
         P1[0, :] - point1[0] * P1[2, :],
         point2[1] * P2[2, :] - P2[1, :],
         P2[0, :] - point2[0] * P2[2, :]
         ]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices=False)

    return Vh[3, 0:3] / Vh[3, 3]


# Add Trackers
trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
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


# Loop Macro Func
def LOOP_FUNCTION(FUNC, **kargs):
    print(TAG, LOOP_FUNCTION.__name__, '->', FUNC.__name__, '[', len(kargs), ']')
    # Function Call
    for key, value in kargs.items():
        ret = FUNC(value)
        print(TAG, LOOP_FUNCTION.__name__, key, ret)
