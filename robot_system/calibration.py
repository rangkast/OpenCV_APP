import cv2

from robot_system.resource.definition import *
from robot_system.resource.robot_system_data import *
from robot_system.common import *

TAG = '[CALIBRATION]'


def cal_iqr_func(arr):
    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)

    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    lower_bound = Q1 - outlier_step
    upper_bound = Q3 + outlier_step

    mask = np.where((arr > upper_bound) | (arr < lower_bound))

    if DEBUG > DEBUG_LEVEL.LV_2:
        print(TAG, f"cal_iqr_func!!!!!! lower_bound = {lower_bound} upper_bound ={upper_bound} mask = {mask}")

    return mask


def detect_outliers(blob_array, remove_index_array):
    for blob_data in blob_array:
        if len(blob_data) != 0:
            temp_x = np.array(cal_iqr_func(blob_data))
            for x in temp_x:
                for xx in x:
                    if xx in remove_index_array:
                        continue
                    else:
                        remove_index_array.append(xx)

    remove_index_array.sort()

    if DEBUG > DEBUG_LEVEL.LV_2:
        print(TAG, "detect_outliers remove_index_array", remove_index_array)


def static_rt_func(*args):
    cam_id = args[0][0]
    METHOD, _ = ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_solve_pnp_setting()
    cam_info_id = cam_id % 2 if ROBOT_SYSTEM_DATA[
                                    SYSTEM_SETTING].get_camera_mode() == CAMERA_MODE.DUAL_CAMERA else cam_id

    dev_name = ROBOT_SYSTEM_DATA[CAM_INFO][cam_info_id].get_dev()
    json_file = ''.join(['jsons/' f'{CAM_INFO}/', f'{dev_name}/', 'rt_data_', str(METHOD), '.json'])
    json_data = rw_json_data(READ, json_file, None)
    lr_position = L_CONTROLLER if ROBOT_SYSTEM_DATA[
                                      SYSTEM_SETTING].get_lr_position() == LR_POSITION.LEFT else R_CONTROLLER

    if ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_mode() == MODE.SYSTEM_SETTING_MODE:
        # SYSTEM_SETTING_MODE
        # Set R|T data to json data
        rt_med_data = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id].get_r_t().get_med_r_t()
        l_name = ROBOT_SYSTEM_DATA[CAM_INFO][cam_info_id].get_name()
        set_rt_format = {'rvecs': rt_med_data.rvecs.tolist(),
                         'tvecs': rt_med_data.tvecs.tolist()}
        if json_data != ERROR and json_data is not None:
            if str(cam_id) in json_data:
                detect = 0
                for i, data_i in enumerate(json_data[str(cam_id)]):
                    if l_name in data_i:
                        detect = 1
                        json_data[str(cam_id)][i][l_name][lr_position]['SRT'] = set_rt_format
                if detect == 0:
                    json_data[str(cam_id)].append({l_name: {lr_position: {'SRT': set_rt_format}}})
            else:
                json_data[str(cam_id)] = [{l_name: {lr_position: {'SRT': set_rt_format}}}]
        else:
            print(TAG, 'not exist')
            json_data = OrderedDict()
            json_data[cam_id] = [{ROBOT_SYSTEM_DATA[CAM_INFO][cam_id].get_name(): {
                lr_position: {'SRT': set_rt_format}}}]
        print(TAG, 'saved ', json_file)
        rw_json_data(WRITE, json_file, json_data)

    else:
        # CALIBRATION_MODE
        # Load R|T from json file and set
        # IF stereoCamera mode use two static RT vectors to triangulate points
        if json_data != ERROR and json_data is not None:
            rt_data = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id].get_r_t()

            # maybe cam_id is 0 or 1
            if ROBOT_SYSTEM_DATA[SYSTEM_SETTING].check_functions('robot_animate_tracker') == SUCCESS:
                cam_id = cam_info_id
            if str(cam_id) in json_data:
                l_name = ROBOT_SYSTEM_DATA[CAM_INFO][cam_info_id].get_name()
                for i, data_i in enumerate(json_data[str(cam_id)]):
                    if l_name in data_i:
                        srvec = json_data[str(cam_id)][i][l_name][lr_position]['SRT']['rvecs']
                        stvec = json_data[str(cam_id)][i][l_name][lr_position]['SRT']['tvecs']
                        med_rt = R_T(np.array(srvec, dtype=np.float64),
                                     np.array([stvec[0], stvec[1], stvec[2]], dtype=np.float64))
                        rt_data.set_med_r_t(med_rt)
                        # restore cam_id
                        cam_id = args[0][0]
                        ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id].set_r_t(rt_data)
        else:
            print(TAG, 'ERROR spec file in not Exist')
            return ERROR

    return SUCCESS


def median_blobs(*args):
    cam_id = args[0][0]
    cam_info_id = cam_id % 2 if ROBOT_SYSTEM_DATA[
                                    SYSTEM_SETTING].get_camera_mode() == CAMERA_MODE.DUAL_CAMERA else cam_id

    # median blobs start
    blob_data = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id].get_blob()
    if len(blob_data.get_acc_blobs()) == 0:
        return ERROR

    _, DO_DISTORT = ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_solve_pnp_setting()

    # read data from dataclass
    acc_blobs = blob_data.get_acc_blobs()
    curr_blobs = blob_data.get_curr_blobs()

    acc_blobs_length = len(acc_blobs)
    curr_blobs_length = len(curr_blobs)
    if acc_blobs_length == 0:
        print(TAG, 'acc_blobs_length is 0 ERROR')
        return ERROR

    med_blobs = []
    remove_index_array = []

    if DEBUG > DEBUG_LEVEL.LV_2:
        print(TAG, 'cam_id:', cam_id, 'acc_blobs_length:', acc_blobs_length, 'curr_blobs_length:', curr_blobs_length)

    for c_blob in curr_blobs:
        med_xy = [[], [], []]
        for a_blob in acc_blobs:
            if c_blob.idx == a_blob.idx:
                med_xy[0].append(a_blob.x)
                med_xy[1].append(a_blob.y)
        detect_outliers(med_xy, remove_index_array)

        count = 0
        for index in remove_index_array:
            med_xy[0].pop(index - count)
            med_xy[1].pop(index - count)
            count += 1

        mean_med_x = np.mean(med_xy[0])
        mean_med_y = np.mean(med_xy[1])

        # Change med blob data if UNDISTORTION set
        if DO_DISTORT == UNDISTORTION:
            camera_k, dist_coeff = ROBOT_SYSTEM_DATA[CAM_INFO][cam_info_id].get_camera_calibration()
            mean_med_undistort = cv2.undistortPoints(np.array([[mean_med_x, mean_med_y]], dtype=np.float64),
                                                     camera_k,
                                                     dist_coeff)
            mean_med_x = mean_med_undistort[0][0][0]
            mean_med_y = mean_med_undistort[0][0][1]

        med_blobs.append(BLOB_2D(c_blob.idx, mean_med_x, mean_med_y))

    r_len = len(remove_index_array)
    if DEBUG > DEBUG_LEVEL.LV_2:
        print(TAG, f"median_blobs!!!!! remove_index_array length={r_len}")
        print(TAG, med_blobs)
    blob_data.set_med_blobs(med_blobs)
    ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id].set_blob(blob_data)

    if ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_mode() == MODE.SYSTEM_SETTING_MODE:
        # median R|T start
        rt_data = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id].get_r_t()
        acc_r_t = rt_data.get_acc_r_t()
        acc_row = len(rt_data.get_curr_r_t().rvecs[0])
        # ToDo
        acc_col = 3

        # start of R IQR
        med_rvecs = [[] for i in range(acc_col * acc_row)]
        remove_index_array.clear()
        for c_r_t in acc_r_t:
            pos = 0
            for col in range(acc_col):
                for row in range(acc_row):
                    med_rvecs[pos].append(c_r_t.rvecs[col][row])
                    pos += 1

        detect_outliers(med_rvecs, remove_index_array)
        count = 0
        for index in remove_index_array:
            for med_rvecs_data in med_rvecs:
                med_rvecs_data.pop(index - count)
            count += 1

        med_rvecs_result = [[] for i in range(acc_col)]
        count = 0
        for col in range(acc_col):
            for row in range(acc_row):
                med_rvecs_result[col].append(np.mean(med_rvecs[count]))
                count += 1

        # end of R IQR
        med_tvecs = [[], [], []]
        remove_index_array.clear()

        for c_r_t in acc_r_t:
            med_tvecs[0].append(c_r_t.tvecs[0][0])
            med_tvecs[1].append(c_r_t.tvecs[1][0])
            med_tvecs[2].append(c_r_t.tvecs[2][0])

        detect_outliers(med_tvecs, remove_index_array)
        count = 0
        for index in remove_index_array:
            med_tvecs[0].pop(index - count)
            med_tvecs[1].pop(index - count)
            med_tvecs[2].pop(index - count)
            count += 1

        med_tvecs_x = np.mean(med_tvecs[0])
        med_tvecs_y = np.mean(med_tvecs[1])
        med_tvecs_z = np.mean(med_tvecs[2])

        RVECS = np.array([med_rvecs_result], dtype=np.float64)
        TVECS = np.array([[med_tvecs_x], [med_tvecs_y], [med_tvecs_z]], dtype=np.float64)

        med_rt = R_T(RVECS, TVECS)

        if DEBUG > DEBUG_LEVEL.LV_2:
            print(TAG, 'med_blobs: ', med_blobs)
            print(TAG, 'med_rt: ', med_rt)

        rt_data.set_med_r_t(med_rt)
        ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id].set_r_t(rt_data)

    return SUCCESS


def pair_points(*args):
    cam_id = args[0][0]
    group_num = args[0][1]

    blob_data = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id].get_blob()
    led_cnt = ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_led_cnt()
    if len(blob_data.get_acc_blobs()) == 0:
        return ERROR

    # ToDo
    if str(group_num) not in ROBOT_SYSTEM_DATA[GROUP_DATA_INFO]:
        ROBOT_SYSTEM_DATA[GROUP_DATA_INFO][str(group_num)] = [GROUP_DATA() for i in range(led_cnt)]

    for data in blob_data.get_med_blobs():
        ROBOT_SYSTEM_DATA[GROUP_DATA_INFO][str(group_num)][data.idx].set_pair_xy(
            {'cidx': cam_id, 'group_num': group_num, 'blob': data})

    return SUCCESS


def coordRefactor(cam_l, cam_r):
    METHOD, DO_DISTORT = ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_solve_pnp_setting()
    cam_l_id = cam_l['cidx']
    cam_r_id = cam_r['cidx']
    l_blob = (float(cam_l['blob'].x), float(cam_l['blob'].y))
    r_blob = (float(cam_r['blob'].x), float(cam_r['blob'].y))

    l_r_t = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_l_id].get_r_t().get_med_r_t()
    r_r_t = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_r_id].get_r_t().get_med_r_t()

    if METHOD == POSE_ESTIMATION_METHOD.STEREO_CAMERA:
        triangulation = DLT(l_r_t.rvecs[0], r_r_t.rvecs[0], l_blob, r_blob)
        get_points = [[triangulation]]
    else:
        l_rotation, jacobian = cv2.Rodrigues(l_r_t.rvecs)
        r_rotation, jacobian = cv2.Rodrigues(r_r_t.rvecs)
        if DO_DISTORT == UNDISTORTION:
            l_projection = np.dot(default_cameraK, np.hstack((l_rotation, l_r_t.tvecs)))
            r_projection = np.dot(default_cameraK, np.hstack((r_rotation, r_r_t.tvecs)))
        else:
            if ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_camera_mode() == CAMERA_MODE.DUAL_CAMERA:
                cal_l_id = cam_l_id % 2
                cal_r_id = cam_r_id % 2

            camera_k, dist_coeff = ROBOT_SYSTEM_DATA[CAM_INFO][cal_l_id].get_camera_calibration()
            l_projection = np.dot(camera_k, np.hstack((l_rotation, l_r_t.tvecs)))
            camera_k, dist_coeff = ROBOT_SYSTEM_DATA[CAM_INFO][cal_r_id].get_camera_calibration()
            r_projection = np.dot(camera_k, np.hstack((r_rotation, r_r_t.tvecs)))
        triangulation = cv2.triangulatePoints(l_projection, r_projection, l_blob, r_blob)
        homog_points = triangulation.transpose()
        get_points = cv2.convertPointsFromHomogeneous(homog_points)

    if DEBUG > DEBUG_LEVEL.LV_2:
        print('cam_l_id ', cam_l_id)
        print('l_r_t ', l_r_t)
        print('l_blob ', l_blob)
        print('cam_r_id ', cam_r_id)
        print('r_r_t ', r_r_t)
        print('r_blob ', r_blob)
        print('get_points ', get_points)

    return get_points


def refactor_3d_point():
    print(TAG, refactor_3d_point.__name__)
    for key, value in ROBOT_SYSTEM_DATA[GROUP_DATA_INFO].items():
        for idx, led_data in enumerate(value):
            pair_xy = led_data.get_pair_xy()
            led_pair_cnt = len(pair_xy)
            if led_pair_cnt < 2:
                print(TAG, f'Error LED Num {idx} has no more than 2-cameras')
            else:
                if DEBUG > DEBUG_LEVEL.LV_2:
                    print(TAG, 'led num ', idx, ' ', pair_xy)
                comb_led = list(itertools.combinations(pair_xy, 2))
                for comb_data in comb_led:
                    if DEBUG > DEBUG_LEVEL.DISABLE:
                        print(TAG, 'comb_led idx: ', idx, ' : ', comb_data)
                    result = coordRefactor(comb_data[0], comb_data[1])
                    led_data.set_remake_3d(REMAKE_3D(comb_data[0]['cidx'], comb_data[1]['cidx'],
                                                     BLOB_3D(idx, result[0][0][0], result[0][0][1], result[0][0][2])))


def find_center(frame, led_num, side, width, SPEC_AREA, cam_id):
    x_sum = 0
    t_sum = 0
    y_sum = 0
    g_c_x = 0
    g_c_y = 0
    m_count = 0

    (X, Y, W, H) = SPEC_AREA
    ret_blobs = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id].get_blob()
    ret_curr_blobs = ret_blobs.get_curr_blobs()

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
        return ERROR, NOT_SET

    if side == RIGHT:
        g_c_x -= width

    if DEBUG > DEBUG_LEVEL.LV_2:
        result_data_str = ' x ' + f'{g_c_x}' + ' y ' + f'{g_c_y}'
        print(TAG, 'cam ', cam_id, ' ', led_num,
              " R:" + f'{result_data_str}' if side == RIGHT else "L:" + f'{result_data_str}')

    if len(ret_curr_blobs) > 0:
        detect = 0
        for i, datas in enumerate(ret_curr_blobs):
            led = datas.idx
            if led == led_num:
                ret_curr_blobs[i] = BLOB_2D(led_num, g_c_x, g_c_y)
                detect = 1
                break
        if detect == 0:
            ret_curr_blobs = np.append(ret_curr_blobs, BLOB_2D(led_num, g_c_x, g_c_y))
    else:
        ret_curr_blobs = np.append(ret_curr_blobs, BLOB_2D(led_num, g_c_x, g_c_y))

    ret_blobs.set_acc_blobs(BLOB_2D(led_num, g_c_x, g_c_y))
    ret_blobs.set_curr_blobs(ret_curr_blobs)

    return SUCCESS, ret_blobs


def generate_led_blobs_format(*args):
    cam_id = args[0][0]
    image_points = []

    blob_data = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id].get_blob()
    cam_info_id = cam_id % 2 \
        if ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_camera_mode() == CAMERA_MODE.DUAL_CAMERA \
        else cam_id

    if len(blob_data.get_acc_blobs()) == 0:
        return ERROR

    for blobs in blob_data.get_curr_blobs():
        image_points.append([blobs.x, blobs.y])

    # Get camera calibration data
    camera_k, dist_coeff = ROBOT_SYSTEM_DATA[CAM_INFO][cam_info_id].get_camera_calibration()
    points2D_d = np.array(image_points, dtype=np.float64)
    points2D_u = cv2.undistortPoints(points2D_d, camera_k, dist_coeff)

    blob_data.set_distort_2d(points2D_d)
    blob_data.set_undistort_2d(points2D_u)
    ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id].set_blob(blob_data)
    return SUCCESS


def camera_pose_estimation(*args):
    METHOD, DO_DISTORT = ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_solve_pnp_setting()

    if ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_mode() == MODE.CALIBRATION_MODE and \
            METHOD == POSE_ESTIMATION_METHOD.STEREO_CAMERA:
        return CONTINUE

    cam_id = args[0][0]
    model_points = []

    blob_data = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id].get_blob()
    if len(blob_data.get_acc_blobs()) == 0:
        return ERROR

    cam_info_id = cam_id % 2 \
        if ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_camera_mode() == CAMERA_MODE.DUAL_CAMERA \
        else cam_id
    for blobs in blob_data.get_curr_blobs():
        model_points.append(ROBOT_SYSTEM_DATA[LED_INFO][blobs.idx].get_pos())

    # Get camera calibration data
    camera_k, dist_coeff = ROBOT_SYSTEM_DATA[CAM_INFO][cam_info_id].get_camera_calibration()
    points3D = np.array(model_points, dtype=np.float64)
    points2D_d = blob_data.get_distort_2d()
    points2D_u = blob_data.get_undistort_2d()

    INPUT_ARRAY = [
        cam_id,
        points3D,
        points2D_d if DO_DISTORT == DISTORTION else points2D_u,
        camera_k if DO_DISTORT == DISTORTION else default_cameraK,
        dist_coeff if DO_DISTORT == DISTORTION else default_distCoeff
    ]

    ret, rvecs, tvecs, inliers = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)

    if ret == SUCCESS:
        r_t_data = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id].get_r_t()
        r_t_data.set_curr_r_t(R_T(rvecs, tvecs))
        r_t_data.set_acc_r_t(R_T(rvecs, tvecs))
        ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id].set_r_t(r_t_data)
    elif ret == CONTINUE:
        print('cam id ', cam_id)
        print(rvecs)
        print(tvecs)
        # StereoCalibrate
        if rvecs != NOT_SET and tvecs != NOT_SET:
            # Left Cam
            left_cam_id = cam_id - 1
            r_t_data = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][left_cam_id].get_r_t()
            r_t_data.set_curr_r_t(R_T(rvecs, [[0], [0], [0]]))
            r_t_data.set_acc_r_t(R_T(rvecs, [[0], [0], [0]]))
            ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][left_cam_id].set_r_t(r_t_data)
            # Right Cam
            right_cam_id = cam_id
            r_t_data = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][right_cam_id].get_r_t()
            r_t_data.set_curr_r_t(R_T(tvecs, [[0], [0], [0]]))
            r_t_data.set_acc_r_t(R_T(tvecs, [[0], [0], [0]]))
            ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][right_cam_id].set_r_t(r_t_data)
            ret = SUCCESS
        else:
            return ret

    elif ret == ERROR:

        return ret

    if DEBUG > DEBUG_LEVEL.LV_2:
        RER = reprojection_error(points3D,
                                 points2D_d if DO_DISTORT == DISTORTION else points2D_u,
                                 rvecs, tvecs,
                                 camera_k if DO_DISTORT == DISTORTION else default_cameraK,
                                 dist_coeff if DO_DISTORT == DISTORTION else default_distCoeff)
        print(TAG, 'METHOD ', METHOD, ' DO_DISTORT ', DO_DISTORT)
        print(TAG, 'points3D')
        print(points3D)
        print(TAG, 'points2D_d')
        print(points2D_d)
        print(TAG, 'points2D_u')
        print(points2D_u)
        print(TAG, 'rvecs ', rvecs)
        print(TAG, 'tvecs ', tvecs)
        print(TAG, 'RER ', RER)

    return ret


def reprojection_error(points3D, points2D, rvec, tvec, camera_k, dist_coeff):
    points2D_reprojection, _ = cv2.projectPoints(points3D, rvec, tvec, camera_k, dist_coeff)
    RER = np.average(np.linalg.norm(points2D - points2D_reprojection, axis=(1, 2)))
    return RER


# Add Camera Pose estimation API here

# Default solvePnPRansac
def solvepnp_ransac(*args):
    cam_id = args[0][0]
    points3D = args[0][1]
    points2D = args[0][2]
    camera_k = args[0][3]
    dist_coeff = args[0][4]
    # check assertion
    if len(points3D) != len(points2D):
        print(TAG, "assertion len is not equal")
        return ERROR, NOT_SET, NOT_SET, NOT_SET

    if len(points2D) < 4:
        print(TAG, "assertion < 4: ")
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
            print(TAG, 'solvePnPRefineLM requires OpenCV >= 4.1.1, skipping refinement')
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
        print(TAG, "assertion len is not equal")
        return ERROR, NOT_SET, NOT_SET, NOT_SET

    if len(points2D) < 3 or len(points2D) > 4:
        print(TAG, "assertion ", len(points2D))
        return ERROR, NOT_SET, NOT_SET, NOT_SET

    ret, rvecs, tvecs = cv2.solvePnP(points3D, points2D,
                                     camera_k,
                                     dist_coeff,
                                     flags=cv2.SOLVEPNP_AP3P)

    return SUCCESS if ret == True else ERROR, rvecs, tvecs, NOT_SET


# ToDo
# solvePnPRansac + Essential Matrix + recoverPose, 6 points need
def solvepnp_ransac_recoverPose(*args):
    METHOD, DO_DISTORT = ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_solve_pnp_setting()
    cam_id = args[0][0]
    if cam_id % 2 == 0:
        return CONTINUE, NOT_SET, NOT_SET, NOT_SET
    cam_info_id = (cam_id - 1) % 2 \
        if ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_camera_mode() == CAMERA_MODE.DUAL_CAMERA \
        else (cam_id - 1)
    points3D = args[0][1]
    points2D_L = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id - 1].get_blob().get_distort_2d() \
        if DO_DISTORT == DISTORTION \
        else ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id - 1].get_blob().get_undistort_2d()

    points2D_R = args[0][2]
    camera_k_R = args[0][3]
    dist_coeff_R = args[0][4]
    camera_k_L, dist_coeff_L = ROBOT_SYSTEM_DATA[CAM_INFO][cam_info_id].get_camera_calibration() \
                                   if DO_DISTORT == DISTORTION \
                                   else default_cameraK, default_distCoeff

    # obj_points = []
    # img_points_l = []
    # img_points_r = []
    # length = len(points3D)
    # objectpoint = np.zeros((length, D3D), np.float32)
    # imgpointl = np.zeros((length, D2D), np.float32)
    # imgpointr = np.zeros((length, D2D), np.float32)
    # for idx, obj_data in enumerate(points3D):
    #     objectpoint[idx] = obj_data
    #     imgpointl[idx] = points2D_L[idx]
    #     imgpointr[idx] = points2D_R[idx]
    # obj_points.append(objectpoint)
    # img_points_l.append(imgpointl)
    # img_points_r.append(imgpointr)

    img_points_l = points2D_L.ravel().reshape(len(points3D), D2D)
    img_points_r = points2D_R.ravel().reshape(len(points3D), D2D)

    print('points3D', points3D)
    print('points2D_L', img_points_l)
    print('points2D_R', img_points_r)

    E, mask = cv2.findEssentialMat(img_points_l, img_points_r, 1.0, (0, 0),
                                   cv2.RANSAC, 0.999, 1, None)


    print('E', E)

    # Get Essential Matrix
    # Calculation recoverPose

    return ERROR, NOT_SET, NOT_SET, NOT_SET


def stereo_calibrate(*args):
    METHOD, DO_DISTORT = ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_solve_pnp_setting()
    cam_id = args[0][0]
    if cam_id % 2 == 0:
        return CONTINUE, NOT_SET, NOT_SET, NOT_SET
    cam_info_id = (cam_id - 1) % 2 \
        if ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_camera_mode() == CAMERA_MODE.DUAL_CAMERA \
        else (cam_id - 1)
    points3D = args[0][1]
    points2D_L = ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id - 1].get_blob().get_distort_2d() \
        if DO_DISTORT == DISTORTION \
        else ROBOT_SYSTEM_DATA[MEASUREMENT_INFO][cam_id - 1].get_blob().get_undistort_2d()

    points2D_R = args[0][2]
    camera_k_R = args[0][3]
    dist_coeff_R = args[0][4]
    camera_k_L, dist_coeff_L = ROBOT_SYSTEM_DATA[CAM_INFO][cam_info_id].get_camera_calibration() \
                                   if DO_DISTORT == DISTORTION \
                                   else default_cameraK, default_distCoeff

    obj_points = []
    img_points_l = []
    img_points_r = []
    length = len(points3D)
    objectpoint = np.zeros((length, D3D), np.float32)
    imgpointl = np.zeros((length, D2D), np.float32)
    imgpointr = np.zeros((length, D2D), np.float32)

    for idx, obj_data in enumerate(points3D):
        objectpoint[idx] = obj_data
        imgpointl[idx] = points2D_L[idx]
        imgpointr[idx] = points2D_R[idx]
    obj_points.append(objectpoint)
    img_points_l.append(imgpointl)
    img_points_r.append(imgpointr)

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
    rmse, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(obj_points, img_points_l, img_points_r,
                                                       camera_k_L, dist_coeff_L,
                                                       camera_k_R, dist_coeff_R,
                                                       (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT),
                                                       criteria_stereo,
                                                       flags)
    # RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    # projection matrix for C1
    P1 = camera_k_L @ RT1

    # RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis=-1)
    # projection matrix for C2
    P2 = camera_k_R @ RT2

    if DEBUG > DEBUG_LEVEL.LV_2:
        print(TAG, 'R ', R, '\nT ', T, '\nE ', E, '\nF ', F)
        print(TAG, 'rmse ', rmse, ' P1 ', P1, ' P2 ', P2)
        print(TAG, 'rodriques ', RT2)
        # Test Code for stereoCam with DLT
        for i in range(length):
            l_blob = (float(points2D_L[i][0][0]), float(points2D_L[i][0][1]))
            r_blob = (float(points2D_R[i][0][0]), float(points2D_R[i][0][1]))
            triangulation = DLT(P1, P2, l_blob, r_blob)
            print(triangulation)

    return CONTINUE if rmse != 0 else ERROR, P1, P2, NOT_SET


SOLVE_PNP_FUNCTION = {
    POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC: solvepnp_ransac,
    POSE_ESTIMATION_METHOD.SOLVE_PNP_REFINE_LM: solvepnp_ransac_refineLM,
    POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P: solvepnp_AP3P,
    POSE_ESTIMATION_METHOD.SOLVE_PNP_RECOVER_POSE: solvepnp_ransac_recoverPose,
    POSE_ESTIMATION_METHOD.STEREO_CAMERA: stereo_calibrate,
}
