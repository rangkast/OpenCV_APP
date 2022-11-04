import re
from collections import OrderedDict

import cv2
from cal_search import *
from cal_def import *
import json


def check_facing_dot(target, cam_pose):
    pts_facing = []
    pts_cam = []

    for data in target:
        # 카메라 pose 기준으로 led 좌표 변환
        idx = int(data['idx'])
        temp = transfer_point(vector3(data['pos'][0], data['pos'][1], data['pos'][2]), cam_pose)
        # led의 방향 벡터 변환
        ori = rotate_point(vector3(data['dir'][0], data['dir'][1], data['dir'][2]), cam_pose)

        _pos_trans = list(map(float, [temp.x, temp.y, temp.z]))
        # 단위 벡터 생성
        normal = nomalize_point(vector3(temp.x, temp.y, temp.z))

        _dir_trans = list(map(float, [ori.x, ori.y, ori.z]))

        # facing dot 찾기/
        facing_dot = get_dot_point(normal, ori)
        angle = math.radians(180.0 - angle_spec)
        rad = np.cos(angle)

        if facing_dot < rad:
            pts_facing.append({'idx': idx, 'pos': list(map(float, [data['pos'][0], data['pos'][1], data['pos'][2]])),
                               'dir': list(map(float, [data['dir'][0], data['dir'][1], data['dir'][2]])),
                               'pattern': data['pattern']})
            pts_cam.append({'idx': cam_pose['idx'], 'pos': _pos_trans, 'dir': _dir_trans, 'pattern': data['pattern']})

    return pts_facing, pts_cam


# end check_facing_dot()

def read_led_pts(fname):
    pts = []
    with open(f'{fname}.txt', 'r') as F:
        a = F.readlines()
    for idx, x in enumerate(a):
        m = re.match(
            '\{ \.pos *= \{+ *(-*\d+.\d+),(-*\d+.\d+),(-*\d+.\d+) *\}+, \.dir *=\{+ *(-*\d+.\d+),(-*\d+.\d+),(-*\d+.\d+) *\}+, \.pattern=(\d+) },',
            x)
        x = float(m.group(1))
        y = float(m.group(2))
        z = float(m.group(3))
        u = float(m.group(4))
        v = float(m.group(5))
        w = float(m.group(6))
        _pos = list(map(float, [x, y, z]))
        _dir = list(map(float, [u, v, w]))
        pts.append({'idx': m.group(7), 'pos': _pos, 'dir': _dir, 'pattern': m.group(7)})

    print(f'{fname} PointsRead')
    leds_dic[fname] = pts


# end read_led_pts_noise()

def cam_projection(idx, var, target, cam_pose, ax, color, calc):
    rmse = 0.0
    RER = 0.0

    obj_cam_pos_n = np.array(cam_pose['position'])

    rotR = R.from_quat(np.array(cam_pose['orient']))
    Rod, _ = cv2.Rodrigues(rotR.as_matrix())

    oret = cv2.projectPoints(var, Rod, obj_cam_pos_n, cameraK, distCoeff)
    o_xx, o_yy = oret[0].reshape(len(var), 2).transpose()

    ret = cv2.projectPoints(target, Rod, obj_cam_pos_n, cameraK, distCoeff)
    xx, yy = ret[0].reshape(len(target), 2).transpose()

    if calc == 1:
        pts = []
        origin_pts = []
        nidx = [x['idx'] for x in trans_leds_array[idx]['pts_facing']]
        for i in range(len(xx)):
            _opos = list(map(float, [o_xx[i], o_yy[i]]))
            origin_pts.append({'idx': nidx[i], 'pos': _opos, 'reserved': 0})

            _pos = list(map(float, [xx[i], yy[i]]))
            pts.append({'idx': nidx[i], 'pos': _pos, 'reserved': 0})
            rmse += np.power((xx[i] - o_xx[i]), 2) + np.power(yy[i] - o_yy[i], 2)

        RER = round(np.sqrt(rmse) / len(xx), 18)
        print('remake offset RER: ', RER)

        draw_dots(2, origin_pts, ax, 'red')
        draw_dots(2, pts, ax, color)

    max_x = 0
    max_y = 0
    if abs(max(xx)) > max_x: max_x = abs(max(xx))
    if abs(min(xx)) > max_x: max_x = abs(min(xx))
    if abs(max(yy)) > max_y: max_y = abs(max(yy))
    if abs(min(yy)) > max_y: max_y = abs(min(yy))
    dimen = max(max_x, max_y)
    dimen *= 1.1
    ax.set_xlim([500, dimen])
    ax.set_ylim([300, dimen])
    return xx, yy, RER


# end cam_projection()

def print_result_and_refactor(idx, ax, rvecs, tvecs):
    position = np.round_(tvecs.reshape(3), 3).tolist()
    rotation = np.round_(R.from_rotvec(rvecs.reshape(3)).as_quat(), 3).tolist()

    observed_pos = tvecs.reshape(3)
    observed_rot = R.from_rotvec(rvecs.reshape(3)).as_quat()
    observed_pose = {'position': vector3(observed_pos[0], observed_pos[1], observed_pos[2]),
                     'orient': quat(observed_rot[0], observed_rot[1], observed_rot[2], observed_rot[3])}

    remake_offset = pose_apply_inverse(observed_pose, camera_array[idx])
    remake_offset_arr.append({'idx': idx, 'position': remake_offset['position'], 'orient': remake_offset['orient']})

    origin_pts = []
    origin_leds = [[x['pos'][0], x['pos'][1], x['pos'][2]] for x in trans_leds_array[idx]['pts_facing']]

    for lpt in origin_leds:
        pt = vector3(lpt[0], lpt[1], lpt[2])
        temp_pt = transfer_point(pt, remake_offset)
        origin_pts.append([temp_pt.x, temp_pt.y, temp_pt.z])

    noise_leds = [[x['pos'][0], x['pos'][1], x['pos'][2]] for x in trans_noise_leds_array[idx]['pts_facing']]

    xx, yy, RER = cam_projection(idx, np.array(origin_pts), np.array(noise_leds), camera_array[idx], ax, 'blue', 1)

    if RER > 1.0:
        print('cam_id', idx, ' check!!!!! RER: ', RER, ' ori: ', np.round_(get_euler_from_quat('zxy', camera_array[idx]['orient']), 3))
        print(f"Position:{np.round_(position, 12)}Rotation:{np.round_(rotation, 12)}")
        print(f"Offset:Pos[{np.round_(remake_offset['position'], 12)}]Orient[{np.round_(remake_offset['orient'], 12)}]")

    return RER


# end print_result_and_refactor()

def print_remake_offset():
    for remake_offset in remake_offset_arr:
        print(
            f"original base !!cam num = {remake_offset['idx']} r_o position {np.round_(remake_offset['position'], 3)} r_o orient {np.round_(remake_offset['orient'], 3)}")
    plt.style.use('default')
    fig_remake_offset = plt.figure(figsize=(10, 10))
    ax = fig_remake_offset.add_subplot(111, projection='3d')

    # 원점
    ax.scatter(0, 0, 0, marker='o', color='k', s=20)
    for remake_offset in remake_offset_arr:
        ax.scatter(remake_offset['position'].x, remake_offset['position'].y, remake_offset['position'].z, marker='x',
                   color='blue', s=20)
        label = (f"remake_offset[{remake_offset['idx']}]")
        ax.text(remake_offset['position'].x, remake_offset['position'].y, remake_offset['position'].z, label, size=10)

    ax.set_xlim([-0.5, 0.5])
    ax.set_xlabel('X')
    ax.set_ylim([-0.5, 0.5])
    ax.set_ylabel('Y')
    ax.set_zlim([-0.5, 0.5])
    ax.set_zlabel('Z')

    detect_outliers(remake_offset_arr, 1)

    print("remake offset idx = [")
    for remake_offset in remake_offset_arr:
        print(f"{remake_offset['idx']}", end=' ')
    print("]")

    for remake_offset in remake_offset_arr:
        print(
            f"remove outlier base !!cam num = {remake_offset['idx']} r_o position {np.round_(remake_offset['position'], 3)} r_o orient {np.round_(remake_offset['orient'], 3)}")

    mean_sd_func(remake_offset_arr)

    # detect_outliers(remake_offset_arr, 2)

    # print("remake offset idx = [")
    # for remake_offset in remake_offset_arr:
    #     print(f"{remake_offset['idx']}", end=' ')
    # print("]")

    # for remake_offset in remake_offset_arr:
    #     print(f"remove outlier base !!cam num = {remake_offset['idx']} r_o position {np.round_(remake_offset['position'], 3)} r_o orient {np.round_(remake_offset['orient'], 3)}")

    # mean_sd_func(remake_offset_arr)

    plt.show()


def mean_sd_func(df):
    pos_arr = [[], [], []]
    ori_arr = [[], [], [], []]

    extract_element(pos_arr, ori_arr, df)

    result_pos_mean = []
    result_pos_mean.append(np.mean(pos_arr[0]))
    result_pos_mean.append(np.mean(pos_arr[1]))
    result_pos_mean.append(np.mean(pos_arr[2]))
    result_pos_mean.append(np.mean(ori_arr[0]))
    result_pos_mean.append(np.mean(ori_arr[1]))
    result_pos_mean.append(np.mean(ori_arr[2]))
    result_pos_mean.append(np.mean(ori_arr[3]))

    print(f"median value pos : [{np.median(pos_arr[0])},{np.median(pos_arr[1])},{np.median(pos_arr[2])}]")
    print(
        f"median value orient : [{np.median(ori_arr[0])},{np.median(ori_arr[1])},{np.median(ori_arr[2])},{np.median(ori_arr[3])}]")
    print(f"mean result pos = [{result_pos_mean[0]},{result_pos_mean[1]},{result_pos_mean[2]}]")
    print(f"mean result ori =[{result_pos_mean[3]},{result_pos_mean[4]},{result_pos_mean[5]},{result_pos_mean[6]}]")


def extract_element(pos_arr, ori_arr, df):
    for i in range(len(df)):
        pos_arr[0].append(float(df[i]['position'].x))
        pos_arr[1].append(float(df[i]['position'].y))
        pos_arr[2].append(float(df[i]['position'].z))
        ori_arr[0].append(float(df[i]['orient'].x))
        ori_arr[1].append(float(df[i]['orient'].y))
        ori_arr[2].append(float(df[i]['orient'].z))
        ori_arr[3].append(float(df[i]['orient'].w))


def detect_outliers(df, func_num):
    pos_arr = [[], [], []]
    ori_arr = [[], [], [], []]
    extract_element(pos_arr, ori_arr, df)

    pos_mask = []
    get_mask(pos_mask, pos_arr, func_num)

    ori_mask = []
    get_mask(ori_mask, ori_arr, func_num)

    temp_mask = []
    for mask in pos_mask:
        add_index_func(mask, temp_mask)

    for mask in ori_mask:
        add_index_func(mask, temp_mask)

    temp_mask.sort()

    print(f"index base mask (temp_mask)={temp_mask}")

    result_mask = []

    for i in range(len(temp_mask)):
        result_mask.append(df[temp_mask[i]]['idx'])

    print(f"idx base remove_index in remake_offset_arr = {result_mask}")
    count = 0
    for var in temp_mask:
        # print(f"var={var}")
        df.pop(var - count)
        count += 1


def get_mask(mask, arr, f_num):
    for i in range(len(arr)):
        if f_num == 1:
            mask.append(cal_iqr_func(arr[i]))
            print(f"mask[{i}] = {mask[i]}")
        else:
            mask.append(modified_zscore(arr[i]))
            print(f"mask[{i}] = {mask[i]}")


def cal_iqr_func(area_arr):
    Q1 = np.percentile(area_arr, 25)
    Q3 = np.percentile(area_arr, 75)

    IQR = Q3 - Q1

    outlier_step = 1.5 * IQR

    print(f"Q1={Q1} Q3={Q3} IQR={IQR} outlier_step={outlier_step}")

    lower_bound = Q1 - outlier_step
    upper_bound = Q3 + outlier_step

    mask = np.where((area_arr > upper_bound) | (area_arr < lower_bound))

    return mask


def modified_zscore(data, consistency_correction=1.4826):
    median = np.median(data)
    print(f"median value = {median}")

    deviation_from_med = np.array(data) - median

    mad = np.median(np.abs(deviation_from_med))
    mod_zcore = deviation_from_med / (consistency_correction * mad)
    mask = np.where((mod_zcore > 2))

    return mask


def add_index_func(area_mask, temp_mask):
    for x in area_mask:
        for y in x:
            if y in temp_mask:
                continue
            else:
                temp_mask.append(y)


def ransac(idx, leds, ax):
    # origin
    origin_leds = [[x['pos'][0], x['pos'][1], x['pos'][2]] for x in leds['pts_facing']]

    # 원본을 볼 때
    # cam_projection(idx, np.array(origin_leds), camera_array[idx], ax, 'white')
    noise_leds = [[x['pos'][0], x['pos'][1], x['pos'][2]] for x in trans_noise_leds_array[idx]['pts_facing']]

    # 각 카메라에서 노이즈를 봤을 때 어떻게 될까.
    xx, yy, RER = cam_projection(idx, np.array(origin_leds), np.array(noise_leds), camera_array[idx], ax, 'red', 0)
    blobs_2d_noise = np.stack((xx, yy), axis=1)

    # check assertion
    check = len(origin_leds)
    # print('check:', check, ' blobs:', len(blobs_2d_noise))

    if check != len(blobs_2d_noise):
        print("assertion not equal: ", len(blobs_2d_noise))
        ax.set_title(f'assertion error={len(blobs_2d_noise)}')
        return -1

    if check < 4 or len(blobs_2d_noise) < 4:
        print("assertion < 4: ", check)
        ax.set_title(f'assertion error={check}')
        return -1
    interationsCount = 100
    confidence = 0.99
    _, rvecs, tvecs, inliers = cv2.solvePnPRansac(np.array(origin_leds), blobs_2d_noise, cameraK, distCoeff)


    # print('rvecs:', rvecs)
    # print('tvecs:', tvecs)
    oret = cv2.projectPoints(np.array(origin_leds), rvecs, tvecs, cameraK, distCoeff)
    o_xx, o_yy = oret[0].reshape(len(np.array(origin_leds)), 2).transpose()

    ret = cv2.projectPoints(np.array(noise_leds), rvecs, tvecs, cameraK, distCoeff)
    xx, yy = ret[0].reshape(len(np.array(noise_leds)), 2).transpose()


    pts = []
    origin_pts = []
    rmse = 0.0
    nidx = [x['idx'] for x in trans_leds_array[idx]['pts_facing']]
    for i in range(len(xx)):
        _opos = list(map(float, [o_xx[i], o_yy[i]]))
        origin_pts.append({'idx': nidx[i], 'pos': _opos, 'reserved': 0})

        _pos = list(map(float, [xx[i], yy[i]]))
        pts.append({'idx': nidx[i], 'pos': _pos, 'reserved': 0})
        rmse += np.power((xx[i] - o_xx[i]), 2) + np.power(yy[i] - o_yy[i], 2)

    RER = round(np.sqrt(rmse) / len(xx), 18)
    print('solve pnp RER: ', RER)

    RER = print_result_and_refactor(idx, ax, rvecs, tvecs)
    return RER


# end ransac()

def draw_ransac_plot(ax):
    plt.style.use('default')
    plt.rc('xtick', labelsize=5)  # x축 눈금 폰트 크기
    plt.rc('ytick', labelsize=5)
    fig_ransac = plt.figure(figsize=(15, 15))

    rer_array = []
    cam_id = []
    cnt = 0
    normal_rer = []
    for idx, leds in enumerate(trans_leds_array):
        fig_ransac.tight_layout()
        ax_ransac = fig_ransac.add_subplot(4, round((len(camera_array) + 1) / 4), idx + 1)
        RER = float('%0.4f' % ransac(idx, leds, ax_ransac))
        ax_ransac.set_title(f'[{idx}]:'f'{RER}', fontdict={'fontsize': 8, 'fontweight': 'medium'})
        if RER != -1:
            rer_array.append(RER)
            cam_id.append(f'{idx}')
            if RER > 1.0:
                cnt += 1
            else:
                normal_rer.append(RER)

    rer_tmp = 0
    for rer in normal_rer:
        rer_tmp += rer

    avg_rer = rer_tmp / len(normal_rer)
    print('avg: ', avg_rer, ' over cnt: ', cnt)

    plt.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.style.use('default')
    plt.figure(figsize=(15, 10))
    plt.title('RER')
    markers, stemlines, baseline = plt.stem(cam_id, rer_array)
    markers.set_color('red')

# end draw_ransac_plot()


def show_leds():
    plt.style.use('default')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim([-0.5, 0.5])
    ax.set_xlabel('X')
    ax.set_ylim([-0.5, 0.5])
    ax.set_ylabel('Y')
    ax.set_zlim([-0.5, 0.5])
    ax.set_zlabel('Z')
    draw_ransac_plot(ax)


# end show_leds()

# Main start
if __name__ == "__main__":
    # LED data 읽고 처리 부분
    read_led_pts('rift_2')
    read_led_pts('rift_6')
    read_led_pts('to_target_pts')
    read_led_pts('to_pts')
    read_led_pts('before')
    # Camera 위치 별 facing dot searching

    with open('rift6_json', encoding="utf-8") as data_file:
        jdata = json.load(data_file, object_pairs_hook=OrderedDict)

    idx = 0
    for led_num in range(15):
        lcam = jdata[f'{led_num}']['remake']['cam_l']
        lrvecs = jdata[f'{led_num}']['remake']['lrvecs']
        ltvecs = jdata[f'{led_num}']['remake']['ltvecs']

        rcam = jdata[f'{led_num}']['remake']['cam_r']
        rrvecs = jdata[f'{led_num}']['remake']['rrvecs']
        rtvecs = jdata[f'{led_num}']['remake']['rtvecs']

        LRVECS = np.array([[lrvecs[0]], [lrvecs[1]], [lrvecs[2]]], dtype=np.float64)
        LTVECS = np.array([[ltvecs[0]], [ltvecs[1]], [ltvecs[2]]], dtype=np.float64)
        cam_pos = LTVECS.reshape(3)
        cam_ori = R.from_rotvec(LRVECS.reshape(3)).as_quat()

        cam_data = {'idx': idx, 'position': vector3(cam_pos[0], cam_pos[1], cam_pos[2]),
                    'orient': quat(cam_ori[0], cam_ori[1], cam_ori[2], cam_ori[3])}
        print('lcam_id: ', lcam,
              ' ', cam_pos,
              ' ', get_euler_from_quat('zxy', cam_ori), ' q:', quat(cam_ori[0], cam_ori[1], cam_ori[2], cam_ori[3]))
        # camera_array.append(cam_data)

        RRVECS = np.array([[rrvecs[0]], [rrvecs[1]], [rrvecs[2]]], dtype=np.float64)
        RTVECS = np.array([[rtvecs[0]], [rtvecs[1]], [rtvecs[2]]], dtype=np.float64)
        cam_pos = RTVECS.reshape(3)
        cam_ori = R.from_rotvec(RRVECS.reshape(3)).as_quat()
        cam_data = {'idx': idx, 'position': vector3(cam_pos[0], cam_pos[1], cam_pos[2]),
                    'orient': quat(cam_ori[0], cam_ori[1], cam_ori[2], cam_ori[3])}
        print('rcam_id: ', rcam,
              ' ', cam_pos,
              ' ', get_euler_from_quat('zxy', cam_ori), ' q:', quat(cam_ori[0], cam_ori[1], cam_ori[2], cam_ori[3]))

    MAX_DEGREE = 18
    cam_id = 0
    for idx in range(MAX_DEGREE):
        degree = idx * 10
        camera_array.append(
            {'idx': cam_id, 'position': vector3(0.0, 0.0, 0.5), 'orient': get_quat_from_euler('zxy', [0, 45, degree])})
        camera_array.append(
            {'idx': cam_id + 1, 'position': vector3(0.0, 0.0, 0.5), 'orient': get_quat_from_euler('zxy', [0, 45, -(180 - degree)])})
        camera_array.append(
            {'idx': cam_id + 2, 'position': vector3(0.0, 0.0, 0.5), 'orient': get_quat_from_euler('zxy', [0, 55, degree])})
        camera_array.append(
            {'idx': cam_id + 3, 'position': vector3(0.0, 0.0, 0.5), 'orient': get_quat_from_euler('zxy', [0, 55, -(180 - degree)])})
        cam_id += 4

    for idx, cam_pose in enumerate(camera_array):
        pts_facing, pts_cam = check_facing_dot(leds_dic['rift_6'], cam_pose)
        trans_leds_array.append({'pts_facing': pts_facing, 'pts_cam': pts_cam})

    for idx, cam_pose in enumerate(camera_array):
        pts_facing, pts_cam = check_facing_dot(leds_dic['before'], cam_pose)
        trans_noise_leds_array.append({'pts_facing': pts_facing, 'pts_cam': pts_cam})

    print('origin')
    for leds in trans_leds_array:
        print(leds)

    print('target')
    for leds in trans_leds_array:
        print(leds)

    plt.style.use('default')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # draw origin data

    draw_dots(3, leds_dic['rift_6'], ax, 'blue')
    draw_dots(3, leds_dic['before'], ax, 'red')

    if draw_camera_pos == ENABLE:
        cam_1 = 2
        cam_2 = 3
        for idx, cam_pose in enumerate(camera_array):
            if idx == cam_1 or idx == cam_2:
                count = len(trans_leds_array[idx]['pts_facing'])
                new_cam_pose = {'idx': cam_pose['idx'], 'position': cam_pose['position'],
                                'orient': get_quat_from_euler('xyz', [90,
                                                                      0,
                                                                      get_euler_from_quat('zxy', cam_pose['orient'])[2]])}
                new_pt = camtoworld(new_cam_pose['position'], new_cam_pose)
                ax.scatter(new_pt.x, new_pt.y, new_pt.z, marker='x', color='blue', s=20)
                label = (f"{new_cam_pose['idx']} [{count}]")
                ax.text(new_pt.x, new_pt.y, new_pt.z, label, size=5)
                if idx == cam_1:
                    cam_1 += 4
                if idx == cam_2:
                    cam_2 += 4

    # 원점
    ax.scatter(0, 0, 0, marker='o', color='k', s=20)
    ax.set_xlim([-0.2, 0.2])
    ax.set_xlabel('X')
    ax.set_ylim([-0.2, 0.2])
    ax.set_ylabel('Y')
    ax.set_zlim([-0.2, 0.2])
    ax.set_zlabel('Z')
    scale = 1.5
    f = zoom_factory(ax, base_scale=scale)
    draw_ransac_plot(ax)

    plt.show()
