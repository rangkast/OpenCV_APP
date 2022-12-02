import json
import random
import signal
from collections import OrderedDict
from dataclasses import dataclass
import copy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
import subprocess
from operator import itemgetter, attrgetter
import re
import subprocess
import cv2
import traceback
import math

import os
import sys

import glob
import itertools
from scipy.spatial import distance


@dataclass
class vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __array__(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def clamp(self, mmin, mmax):
        self.x = max(mmin, self.x)
        self.x = min(mmax, self.x)
        self.y = max(mmin, self.y)
        self.y = min(mmax, self.y)
        self.z = max(mmin, self.z)
        self.z = min(mmax, self.z)

    def mult_me(self, d):
        self.x *= d
        self.y *= d
        self.z *= d

    def normalize_me(self):
        if self.x == 0 and self.y == 0 and self.z == 0:
            return
        len = self.get_length()
        # print('1: ', self.x, ' ', self.y, ' ', self.z, ' ', len)
        self.x /= len
        self.y /= len
        self.z /= len
        # print('2: ', self.x, ' ', self.y, ' ', self.z, ' ', len)
        return vector3(self.x, self.y, self.z)

    def get_length(self):
        return np.sqrt(np.sum(np.power(np.array(self), 2)))

    def copy(self):
        return vector3(self.x, self.y, self.z)

    def round(self, d):
        self.x = round(self.x, d)
        self.y = round(self.y, d)
        self.z = round(self.z, d)

    def round_(self, d):
        return vector3(round(self.x, d), round(self.y, d), round(self.z, d))

    def get_rotated(self, tq):
        q = quat(self.x * tq.w + self.z * tq.y - self.y * tq.z,
                 self.y * tq.w + self.x * tq.z - self.z * tq.x,
                 self.z * tq.w + self.y * tq.x - self.x * tq.y,
                 self.x * tq.x + self.y * tq.y + self.z * tq.z)
        return vector3(tq.w * q.x + tq.x * q.w + tq.y * q.z - tq.z * q.y,
                       tq.w * q.y + tq.y * q.w + tq.z * q.x - tq.x * q.z,
                       tq.w * q.z + tq.z * q.w + tq.x * q.y - tq.y * q.x)

    def add_vector3(self, t):
        return vector3(round(self.x + t.x, 8), round(self.y + t.y, 8), round(self.z + t.z, 8))

    def sub_vector3(self, t):
        return vector3(round(self.x - t.x, 8), round(self.y - t.y, 8), round(self.z - t.z, 8))

    def get_dot(self, t):
        return self.x * t.x + self.y * t.y + self.z * t.z


@dataclass
class quat:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def __array__(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w])

    def mult_me_quat(self, q):
        tmp = self.copy()
        self.x = tmp.w * q.x + tmp.x * q.w + tmp.y * q.z - tmp.z * q.y
        self.y = tmp.w * q.y - tmp.x * q.z + tmp.y * q.w + tmp.z * q.x
        self.z = tmp.w * q.z + tmp.x * q.y - tmp.y * q.x + tmp.z * q.w
        self.w = tmp.w * q.w - tmp.x * q.x - tmp.y * q.y - tmp.z * q.z

    def mult_quat(self, q):
        tmp = self.copy()
        return quat(tmp.w * q.x + tmp.x * q.w + tmp.y * q.z - tmp.z * q.y,
                    tmp.w * q.y - tmp.x * q.z + tmp.y * q.w + tmp.z * q.x,
                    tmp.w * q.z + tmp.x * q.y - tmp.y * q.x + tmp.z * q.w,
                    tmp.w * q.w - tmp.x * q.x - tmp.y * q.y - tmp.z * q.z)

    def copy(self):
        return quat(self.x, self.y, self.z, self.w)

    '''
    def inverse(self):
        dot = quat_get_dot(self, self)
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        self.mult_me(1 / dot)
    '''

    def round(self, d):
        self.x = round(self.x, d)
        self.y = round(self.y, d)
        self.z = round(self.z, d)
        self.w = round(self.w, d)

    def round_(self, d):
        return quat(round(self.x, d), round(self.y, d), round(self.z, d), round(self.w, d))

    def inverse(self):
        dot = quat_get_dot(self, self)
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        self.x = self.x / dot
        self.y = self.y / dot
        self.z = self.z / dot
        self.w = self.w / dot


def quat_get_dot(self, t):
    return self.x * t.x + self.y * t.y + self.z * t.z + self.w * t.w


def get_dot_point(pt, t):
    return pt.get_dot(t)


def nomalize_point(pt):
    return pt.normalize_me()


def rotate_point(pt, pose):
    return pt.get_rotated(pose['orient'])


def transfer_point(pt, pose):
    r_pt = pt.get_rotated(pose['orient'])
    return r_pt.add_vector3(pose['position'])


def move_point(pt, pose):
    return pt.add_vector3(pose)


def transfer_point_inverse(pt, pose):
    t = copy.deepcopy(pose)
    t['orient'].inverse()
    r_pt = pt.sub_vector3(t['position'])
    return r_pt.get_rotated(t['orient'])


def get_quat_from_euler(order, value):
    rt = R.from_euler(order, value, degrees=True)
    return quat(rt.as_quat()[0], rt.as_quat()[1], rt.as_quat()[2], rt.as_quat()[3])


def pose_apply(a, b):
    return {'position': transfer_point(a['position'], b), 'orient': b['orient'].mult_quat(a['orient'])}


def pose_apply_inverse(a, b):
    t = copy.deepcopy(b)
    t['orient'].inverse()
    tmp = a['position'].sub_vector3(t['position'])
    return {'position': tmp.get_rotated(t['orient']), 'orient': t['orient'].mult_quat(a['orient'])}


def get_euler_from_quat(order, q):
    rt = R.from_quat(np.array(q))
    return rt.as_euler(order, degrees=True)


def unit_vector(vector):
    """Returnstheunitvectorofthevector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returnstheangleinradiansbetweenvectors'v1'and'v2'::
    >>>angle_between((1,0,0),(0,1,0))
    1.5707963267948966
    >>>angle_between((1,0,0),(1,0,0))
    0.0
    >>>angle_between((1,0,0),(-1,0,0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# add common function here
def draw_dots(dimension, pts, ax, c):
    if dimension == D3D:
        x = [x['pos'][0] for x in pts]
        y = [x['pos'][1] for x in pts]
        z = [x['pos'][2] for x in pts]
        u = [x['dir'][0] for x in pts]
        v = [x['dir'][1] for x in pts]
        w = [x['dir'][2] for x in pts]
        idx = [x['idx'] for x in pts]
        ax.scatter(x, y, z, marker='o', s=3, color=c, alpha=0.5)
        ax.quiver(x, y, z, u, v, w, length=0.05, linewidths=0.1, color='red', normalize=True)
        for idx, x, y, z in zip(idx, x, y, z):
            label = '%s' % idx
            ax.text(x, y, z, label, size=5)
    elif dimension == D3DT:
        for i in range(len(pts)):
            if pts[i]['remake_3d'] != 'error':
                print(pts[i]['remake_3d'])
                x = [coord['coord'][0][0][0] for coord in pts[i]['remake_3d']]
                y = [coord['coord'][0][0][1] for coord in pts[i]['remake_3d']]
                z = [coord['coord'][0][0][2] for coord in pts[i]['remake_3d']]
                idx = [coord['idx'] for coord in pts[i]['remake_3d']]

                ax.scatter(x, y, z, marker='o', s=3, color=c, alpha=0.5)
                for idx, x, y, z in zip(idx, x, y, z):
                    label = '%s' % idx
                    ax.text(x, y, z, label, size=3)


    elif dimension == D2D:
        x = [x['pos'][0] for x in pts]
        y = [x['pos'][1] for x in pts]
        idx = [x['idx'] for x in pts]
        ax.scatter(x, y, marker='o', s=15, color=c, alpha=0.5)
        # 좌표 numbering
        for idx, x, y in zip(idx, x, y):
            label = '%s' % idx
            ax.text(x + 0.001, y + 0.001, label, size=6)


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
        print('exception')
        return ERROR


def init_model_json(cam_dev_list):
    print('start ', init_model_json.__name__)
    camera_info_array = []
    try:
        A = []
        K = []
        RVECS = []
        TVECS = []
        global group_num
        if DEBUG == ENABLE:
            print(cam_dev_list, ' count:', len(cam_dev_list))
        for i in range(len(cam_dev_list)):
            print('\n')
            print('cam_id[', i, '] :', cam_json[i])
            jsonObject = json.load(open(''.join(['jsons/', f'{LR_POSITION}/', f'{cam_json[i]}'])))
            model_points = jsonObject.get('model_points')
            model = []
            for data in model_points:
                idx = data.split('LED')[1]
                # ToDo
                x = model_points.get(data)[0]
                y = model_points.get(data)[1]
                s = model_points.get(data)[2]
                model.append({'idx': idx, 'x': x, 'y': y, 'size': s})
            cam_info = cam_dev_list[i].split('\n\t')

            '''                        
              k = [ k₁ k₂, k₃, k4 ] for CV1 fisheye distortion                    

                  ⎡ fx 0  cx ⎤
              A = ⎢ 0  fy cy ⎥
                  ⎣ 0  0  1  ⎦            
            '''

            if USE_LIBUSB == ENABLE:
                # ToDo
                print('read calibration data from flash memory')
                init_openhmd_driver()
            else:
                f = jsonObject.get('camera_f')
                c = jsonObject.get('camera_c')
                k = jsonObject.get('camera_k')

                A = np.array([[f[0], 0.0, c[0]],
                              [0.0, f[1], c[1]],
                              [0.0, 0.0, 1.0]], dtype=np.float64)
                K = np.array([[k[0]], [k[1]], [k[2]], [k[3]]], dtype=np.float64)

                if RT_TEST == ENABLE:
                    rvecs = jsonObject.get('rvecs')
                    tvecs = jsonObject.get('tvecs')
                    RVECS = np.array([[rvecs[0]], [rvecs[1]], [rvecs[2]]], dtype=np.float64)
                    TVECS = np.array([[tvecs[0]], [tvecs[1]], [tvecs[2]]], dtype=np.float64)
                else:
                    RVECS = NOT_SET
                    TVECS = NOT_SET

            if DEBUG == ENABLE:
                print('cameraK: ', A)
                print('dist_coeff: ', K)
                if RT_TEST == ENABLE:
                    print('rvecs:', RVECS)
                    print('tvecs:', TVECS)
            if i % 2 == 0:
                group_num = i
            camera_info_array.append({'idx': i,
                                      'name': cam_info[0],
                                      'port': cam_info[1],
                                      'json': cam_json[i],
                                      'group': group_num,
                                      'cam_cal': {'cameraK': A, 'dist_coeff': K},
                                      'model': {'spec': model},
                                      'blobs': [],
                                      'med_blobs': [],
                                      'inliers_loop': [],
                                      'distorted_2d': [],
                                      'undistorted_2d': [],
                                      'detect_status': [NOT_SET, 0, 0],
                                      'track_cal': {'data': [], 'recording': []},
                                      'S_R_T': {'rvecs': RVECS, 'tvecs': TVECS},
                                      'D_R_T': {'rvecs': NOT_SET, 'tvecs': NOT_SET},
                                      'D_R_T_A': [],
                                      'filter': {'status': NOT_SET, 'filter_array': [], 'lm_array': [], 'm_array': []},
                                      'RER': {'loop': RER_MAX, 'prev': RER_MAX, 'curr': RER_MAX, 'min': RER_MAX,
                                              'origin': RER_MAX,
                                              'remake': RER_MAX,
                                              'C_R_T': {'rvecs': RVECS, 'tvecs': TVECS}}})
    except:
        leds_dic['test_status']['json_2'] = ERROR
        print('exception')
        traceback.print_exc()
    else:
        leds_dic['test_status']['json_2'] = DONE
        print('done')
    finally:
        if DEBUG == ENABLE:
            print(leds_dic['test_status'])
    return camera_info_array


def init_coord_json(file):
    print('start ', init_coord_json.__name__)
    try:
        json_file = open(''.join(['jsons/specs/', f'{file}']))
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
            pts[int(idx)] = {'idx': idx,
                             'pos': [x, y, z],
                             'dir': [u, v, w],
                             'res': [r1, r2, r3],
                             'pair_xy': [],
                             'remake_3d': [],
                             'loop_pos': {'rx': NOT_SET, 'ry': NOT_SET, 'rz': NOT_SET,
                                          'dx': NOT_SET, 'dy': NOT_SET, 'dz': NOT_SET,
                                          'dis': RER_MAX,
                                          'remake': NOT_SET}}

            print(''.join(['{ .pos = {{', f'{x}', ',', f'{y}', ',', f'{z}',
                           ' }}, .dir={{', f'{u}', ',', f'{v}', ',', f'{w}', ' }}, .pattern=', f'{idx}', '},']))
    except:
        leds_dic['test_status']['json_1'] = ERROR
        print('exception')
        traceback.print_exc()
    else:
        leds_dic['test_status']['json_1'] = DONE
        print('done')
    finally:
        if DEBUG == ENABLE:
            print(leds_dic['test_status'])
    return pts


def init_test_status():
    print('start ', init_test_status.__name__)
    test_status = {'json_1': NOT_SET,
                   'json_2': NOT_SET,
                   'cam_capture': ERROR,
                   'ransac': ERROR,
                   'triangle': NOT_SET,
                   'event_status': NOT_SET,
                   'bt_status': {'mac': NOT_SET, 'name': NOT_SET, 'list': [], 'cal': []}}
    return test_status


def check_success_adder(target_data):
    str_data = str(target_data)
    leds_dic['test_status'][str_data] += 1

    return leds_dic['test_status'][str_data]


def sudo_cmd(password, command):
    print('start ', sudo_cmd.__name__)
    # bluescan = os.system('echo %s|sudo -S %s' % (password, command))
    (status, result) = subprocess.getstatusoutput('echo %s| sudo -S %s' % (password, command))
    if status == 0:
        return ERROR
    for info in result.split('\n'):
        ble_info = info.split(' ')
        if "RIFT_S" in ble_info[1]:
            print('detected:', ble_info)
            return ble_info
        else:
            print('waiting...')
            continue
    return


def terminal_cmd(cmd_m, cmd_s):
    print('start ', terminal_cmd.__name__)
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
        print('exception')
        traceback.print_exc()
    else:
        print('done')
    finally:
        if DEBUG == ENABLE:
            print(devices)
    temp = []
    temp = result.split('\n\n')
    Rift_Sensor = "Rift Sensor"
    print("==================================================")
    ret_val = []
    for i in range(len(temp)):
        if Rift_Sensor in temp[i]:
            ret_val.append(temp[i])
            print("add list rift_sensor", temp[i])
        else:
            print("skipping camera", temp[i])
    print("==================================================")
    return ret_val


def print_dictionary_data(array):
    print('\n')
    for key, value in array.items():
        print(key, value)
    print('\n')


def linear_algo(target, pts, dimension, adder):
    print('linear algebra start, ', dimension)
    Pc = []
    led_idx = []
    origin_pts = []
    before_pts = []
    after_pts = []

    if dimension == D3D:
        for i in range(len(pts)):
            if pts[i]['remake_3d'] != 'error':
                led_idx.append(i)

        print('led_idx ', led_idx)

        for led_num in led_idx:
            if adder == ENABLE:
                rx = pts[led_num]['remake_3d'][0]['coord'][0][0][0] + leds_dic['pts'][led_num]['pos'][0] * 1
                ry = pts[led_num]['remake_3d'][0]['coord'][0][0][1] + leds_dic['pts'][led_num]['pos'][1] * 1
                rz = pts[led_num]['remake_3d'][0]['coord'][0][0][2] + leds_dic['pts'][led_num]['pos'][2] * 1
            else:
                rx = pts[led_num]['remake_3d'][0]['coord'][0][0][0]
                ry = pts[led_num]['remake_3d'][0]['coord'][0][0][1]
                rz = pts[led_num]['remake_3d'][0]['coord'][0][0][2]
            before_pts.append({'idx': led_num,
                               'pos': [rx, ry, rz],
                               'dir': [target[led_num]['dir'][0],
                                       target[led_num]['dir'][1],
                                       target[led_num]['dir'][2]]})
            Pc.append([rx, ry, rz])
            # ToDo
            origin_pts.append(leds_dic['target_pts'][led_num]['pos'])

        for data in Pc:
            print(data)

        A = np.vstack([np.array(Pc).T, np.ones(len(Pc))]).T
        Rt = np.linalg.lstsq(A, np.array(origin_pts), rcond=None)[0]
        print("Rt=\n", Rt.T)
        # Pd_est : Pa 예측값 (Calibration 값)
        Pd_est = np.matmul(Rt.T, A.T)

        print("Pd_est=\n", Pd_est)
        Err = np.array(origin_pts) - Pd_est.T
        print("Err=\n", Err)

        for i, led_num in enumerate(led_idx):
            after_pts.append({'idx': led_num,
                              'pos': Pd_est.T[i],
                              'dir': [target[led_num]['dir'][0],
                                      target[led_num]['dir'][1],
                                      target[led_num]['dir'][2]]})

        print('origin_pts: ', origin_pts)
        print('before pts: ', before_pts)
        print('after pts: ', after_pts)
        return before_pts, after_pts

    elif dimension == D2D:

        print('before pts: ', before_pts)
        print('after pts: ', after_pts)
        return before_pts, after_pts


def cam_projection(idx_array, var, pos, ori, cameraK, distCoeff, ax, color):
    obj_cam_pos_n = np.array(pos)
    rotR = R.from_quat(np.array(ori))
    Rod, _ = cv2.Rodrigues(rotR.as_matrix())

    ret = cv2.fisheye.projectPoints(var, Rod, obj_cam_pos_n, cameraK, distCoeff)
    xx, yy = ret[0].reshape(len(var), 2).transpose()

    pts = []
    for i in range(len(xx)):
        _pos = list(map(float, [xx[i], yy[i]]))
        pts.append({'idx': idx_array[i], 'pos': _pos, 'reserved': 0})

    draw_dots(D2D, pts, ax, color)

    max_x = 0
    max_y = 0
    if abs(max(xx)) > max_x: max_x = abs(max(xx))
    if abs(min(xx)) > max_x: max_x = abs(min(xx))
    if abs(max(yy)) > max_y: max_y = abs(max(yy))
    if abs(min(yy)) > max_y: max_y = abs(min(yy))
    dimen = max(max_x, max_y)
    dimen *= 1.1
    ax.set_xlim([500, dimen])
    ax.set_ylim([200, dimen])

    # ax.set_xlim([0, 1280])
    # ax.set_ylim([0, 960])

    return xx, yy


def test_compare_result(title, compare, before, after, after_2):
    print('\n\n### ', title, ' ###')
    if DEBUG == ENABLE:
        led_distance = []
        origin_vs_target = []
        rt_led_distance = []
        rt_led_distance_2 = []
        led_num = []

        for i in range(len(compare)):
            idx = int(before[i]['idx'])
            led_num.append(f'LED {idx}')
            print('led num: ', idx)
            print('origin: ', compare[i])
            print('remake: ', before[i]['pos'])
            diff_x = '%0.12f' % (
                    compare[i][0] - before[i]['pos'][0])
            diff_y = '%0.12f' % (
                    compare[i][1] - before[i]['pos'][1])
            diff_z = '%0.12f' % (
                    compare[i][2] - before[i]['pos'][2])
            print('B1:[', diff_x, ',', diff_y, ',', diff_z, ']')
            led_distance.append(np.sqrt(
                np.power(float(diff_x), 2) + np.power(float(diff_y), 2) + np.power(
                    float(diff_z), 2)))
            #
            # diff_x = '%0.12f' % (
            #         compare[i][0] - leds_dic['pts'][i]['pos'][0])
            # diff_y = '%0.12f' % (
            #         compare[i][1] - leds_dic['pts'][i]['pos'][1])
            # diff_z = '%0.12f' % (
            #         compare[i][2] - leds_dic['pts'][i]['pos'][2])
            # print('B1:[', diff_x, ',', diff_y, ',', diff_z, ']')
            # origin_vs_target.append(np.sqrt(
            #     np.power(float(diff_x), 2) + np.power(float(diff_y), 2) + np.power(
            #         float(diff_z), 2)))

            if RT_TEST == ENABLE:
                rt_test_diff_x = '%0.12f' % (
                        compare[i][0] - after[i]['pos'][0])
                rt_test_diff_y = '%0.12f' % (
                        compare[i][1] - after[i]['pos'][1])
                rt_test_diff_z = '%0.12f' % (
                        compare[i][2] - after[i]['pos'][2])
                print('A1:[', rt_test_diff_x, ',', rt_test_diff_y, ',', rt_test_diff_z, ']')
                rt_led_distance.append(np.sqrt(
                    np.power(float(rt_test_diff_x), 2) + np.power(float(rt_test_diff_y), 2) + np.power(
                        float(rt_test_diff_z), 2)))

                # rt_test_diff_x_2 = '%0.12f' % (
                #         compare[i][0] - after_2[i]['pos'][0])
                # rt_test_diff_y_2 = '%0.12f' % (
                #         compare[i][1] - after_2[i]['pos'][1])
                # rt_test_diff_z_2 = '%0.12f' % (
                #         compare[i][2] - after_2[i]['pos'][2])
                # print('A2:[', rt_test_diff_x_2, ',', rt_test_diff_y_2, ',', rt_test_diff_z_2, ']')
                # rt_led_distance_2.append(np.sqrt(
                #     np.power(float(rt_test_diff_x_2), 2) + np.power(float(rt_test_diff_y_2), 2) + np.power(
                #         float(rt_test_diff_z_2), 2)))

                ################
                leds_dic['after_led'][idx] = {'idx': idx, 'pos': after[i]['pos'], 'dir': before[i]}
                leds_dic['before_led'][idx] = {'idx': idx, 'pos': before[i]['pos'], 'dir': before[i]}
                #################

            print('\n')

        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        plt.style.use('default')
        plt.figure(figsize=(15, 10))
        plt.title(title)
        markers, stemlines, baseline = plt.stem(led_num, led_distance, label='Before LSM')
        # markers.set_color('red')
        # markers, stemlines, baseline = plt.stem(led_num, origin_vs_target, label='origin - target')
        # markers.set_color('green')

        if RT_TEST == ENABLE:
            markers, stemlines, baseline = plt.stem(led_num, rt_led_distance, label='After LSM')
            markers.set_color('black')
            # markers, stemlines, baseline = plt.stem(led_num, rt_led_distance_2, label='Add origin')
            # markers.set_color('blue')
        plt.legend()


def lsm_and_print_result(title, target, data):
    if leds_dic['test_status']['cam_capture'] == NOT_SET:
        return ERROR

    # Dynamic R|T로 복원한 것을 pts에 lsm
    before, after = linear_algo(target, data, D3D, DISABLE)
    before_2, after_2 = linear_algo(target, data, D3D, ENABLE)

    target_pts_array = []

    # sample devie에 붙이는 게 목표
    for leds in target:
        idx = int(leds['idx'])
        target_pts_array.append(data[idx]['pos'])
    test_compare_result(title, target_pts_array, before, after, after_2)


def draw_ax_plot():
    plt.style.use('default')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # draw origin data
    draw_dots(D3D, leds_dic['target_pts'], ax, 'blue')
    draw_dots(D3DT, leds_dic['target_pts'], ax, 'gray')

    # Add refactoring data

    # 원점
    ax.scatter(0, 0, 0, marker='o', color='k', s=20)
    ax.set_xlim([-0.5, 0.5])
    ax.set_xlabel('X')
    ax.set_ylim([-0.5, 0.5])
    ax.set_ylabel('Y')
    ax.set_zlim([-0.5, 0.5])
    ax.set_zlabel('Z')
    scale = 1.5
    f = zoom_factory(ax, base_scale=scale)
    plt.show()


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


def cal_RER_px(led_ids, frame, points3D, points2D, inliers, rvecs, tvecs, camera_k, dist_coeff, status):
    # Compute re-projection error.
    blob_array = []
    points2D_reproj = cv2.projectPoints(points3D, rvecs,
                                        tvecs, camera_k, dist_coeff)[0].squeeze(1)
    # print('points2D_reproj\n', points2D_reproj, '\npoints2D\n', points2D, '\n inliers: ', inliers)
    assert (points2D_reproj.shape == points2D.shape)
    error = (points2D_reproj - points2D)[inliers]  # Compute error only over inliers.
    # print('error', error)
    rmse = 0.0
    dis = 0.0
    led_except = -1
    except_inlier = -1
    led_dis = []

    for idx, error_data in enumerate(error[:, 0]):
        rmse += np.power(error_data[0], 2) + np.power(error_data[1], 2)
        temp_dis = np.power(error_data[0], 2) + np.power(error_data[1], 2)
        led_dis.append(temp_dis)

        if status == NOT_SET:
            if temp_dis > dis:
                dis = temp_dis
                led_except = led_ids[idx]
                except_inlier = idx

        # print('led_num: ', led_ids[idx], ' dis:', '%0.18f' % temp_dis, ' : ',
        #       points2D_reproj[idx][0], ' ', points2D_reproj[idx][1],
        #       ' vs ', points2D[idx][0], ' ', points2D[idx][1])

        if temp_dis > 100:
            return ERROR, 0, 0, blob_array, except_inlier

    trmse = float(rmse - dis)
    # print('trmse : ', trmse, ' rmse : ', rmse)
    if inliers is None:
        return ERROR, -1, -1, blob_array, except_inlier
    RER = round(np.sqrt(rmse) / len(inliers), 18)
    if status == NOT_SET:
        TRER = round(np.sqrt(trmse) / (len(inliers) - 1), 18)
        if led_except == -1:
            return ERROR, RER, TRER, blob_array, except_inlier
    else:
        TRER = round(np.sqrt(trmse) / (len(inliers)), 18)

    for i, idx in enumerate(led_ids):
        if idx != led_except:
            blob_array.append({'idx': led_ids[i],
                               'cx': points2D[i][0], 'cy': points2D[i][1], 'area': 0})

    return SUCCESS, RER, TRER, blob_array, except_inlier


def getDistance(pointa, pointb):
    xDiff = pointa[0] - pointb[0]
    yDiff = pointa[1] - pointb[1]
    return math.sqrt(xDiff * xDiff + yDiff * yDiff)


########################################### DEFINE #####################################################
'''
    leds_dic['pts']
        - led info dictionary    
        - leds_dic['pts'][led_number] =
            'idx': idx, : led_number 
            'pos': [x, y, z], : model points from rift_s_*.json
            'dir': [u, v, w], : directions from rift_s_*.json
            'res': [r1, r2, r3], : ?? datas from rift_s_*.json
            'pair_xy': [{'cidx': camera_id, 'led_num':led_num, 'cx':cx, 'cy':cy}], : pair 2d coords in same led_num
            'remake_3d': {camera_l, camera_r, coord} or 'error' : candidates 3d coords from 2Dto3D conversion

    leds_dic['cam_info']
        - camera info dictionary
        - leds_dic['cam_info'][camera_number] = 
            'idx': i, : camera number
            'name': cam_info[0], : device name what attached usb serial to linux pc
            'port': cam_info[1], : port number to video
            'cam_cal' : {'cameraK' : A[], 'dist_coeff': K[]}
            'model': {'spec':[]} : 2D boundary spec to set camera position from ***.json
            'blobs': NOT_SET or array, : translated LED id in 2D coord
            'med_blobs': NOT_SET or array, : median blobs array
            'individual_blobs' : NOT_SET or array, : translated LED id in 2D coord with individual RT matrix
            'undistorted_2d': [x,y] : cv2.undistortPoints(list_2d_distorted, camera_k, dist_coeff)
            'detect_status': [DONE or NOT_SET, count], pose checking status and counts
            'S_R_T': {'rvecs': rvecs, 'tvecs': tvecs} : Camera R|T Matrix, Static RT, define in json file
            'D_R_T': {'rvecs': rvecs, 'tvecs': tvecs} : Camera R|T Matrix, Dynamic RT, Calculate RT with solvePnP in frame
            'D_R_T_A': {'rvecs': rvecs, 'tvecs': tvecs} : Camera R|T Matrix, Dynamic RT, Array
            'filter': {'status': DONE OR NOT_SET, 'filter_array': [], 'lm_array':[], 'm_array': []} : filter status
            'RER': {'temp': temp RER, 'origin': med RER, 'remake': final RER}

    leds_dic['test_status']
        - test process status
        - leds_dic['test_status'] =
            'json_1': NOT_SET or DONE, : read model points form rift_s_*.json
            'json_2': NOT_SET or DONE, : read 2d boundary spec from ***.json
            'cam_capture': ERROR, : cam capture success count
            'ransac': ERROR, : do solvePnPRansac
            'triangle': NOT_SET or DONE : do triangulatePoints
            'event_status': {'event': event, 'x': x, 'y': y, 'flag': flag, 'prev_event': prev_event}
            'bt_status': {'mac': MAC address, 'name': Device name, 'lists': [], 'cal': []}
'''
leds_dic = {}

ENABLE = 1
DISABLE = 0

D3DT = 4
D3D = 3
D2D = 2

READ = 0
WRITE = 1

# offset value
offset = {'position': vector3(0.05, 0.05, 0.05), 'orient': get_quat_from_euler('xyz', [6, 6, 15])}
angle_spec = 70

ERROR = -1
SUCCESS = 1

DONE = 'DONE'
NOT_SET = 'NOT_SET'

SUDO_PASSWORD = ''

# CAMERA
pts_refactor = []
pts_noise_refactor = []
cameraK = np.eye(3).astype(np.float64)
distCoeff = np.zeros((4, 1)).astype(np.float64)
cam_count_row = 3
add_pose_offset = ENABLE

DEBUG = ENABLE

MED_MODE = 0
TEMP_MODE = 1
CAL_MODE = 2
QUIVER_MODE = 3
AFTER_MEDIAN_MODE = 4

LOOP_CNT = 1000000

LED_AREA_MIN = 12
LED_AREA_MAX = 300

RER_SPEC = 1.0
RER_MAX = 100
LED_DISTANCE_SPEC = 0.05

CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960

MAX_BLOB_SIZE = 250

CV_FINDCONTOUR_LVL = 140
BLOB_SLOPE_FEATURE = DISABLE
CV_THRESHOLD = 170
CV_MIN_THRESHOLD = 170
CV_MID_THRESHOLD = 190
CV_MAX_THRESHOLD = 255

USE_CUSTOM_FILTER = ENABLE
USE_PRINT_FRAME = DISABLE
PRINT_FRAME_INFOS = DISABLE
USE_LIBUSB = DISABLE

RT_TEST = ENABLE
DYNAMIC_RT = 0
STATIC_RT = 1
DYNAMIC_RT_MED = 2
DYNAMIC_RT_QUIVER = 3
RT_INLIERS = 4
MED_RT = 5

DO_ESTIMATE_POSE = ENABLE
DO_SOLVEPNP_REFINE = DISABLE
DO_UNDISTORT = DISABLE
DO_SOLVEPNP_RANSAC = ENABLE
USE_PRINT_FRAME = DISABLE
PRINT_FRAME_INFOS = DISABLE

MAKE_STATIC_RT = DISABLE

SYSTEM_SETTING_MODE = ENABLE
INLIER_LOOP_SYSTEM = ENABLE

SINGLE_CAMERA_TITLE = 'single camera thread'

LED_COUNT = 15

'''
camera position
'''

# cam_json = ['cam_110.json', 'cam_101.json',
#             'cam_002.json', 'cam_010.json',
#             'cam_004.json', 'cam_005.json',
#             'cam_011.json', 'cam_003.json',
#             'cam_000.json', 'cam_001.json',
#             'cam_111.json', 'cam_100.json',
#             ]
cam_json = ['cam_005.json']
ORIGIN = 'rifts2_left.json'
TARGET = 'rifts2_left.json'
INLIER_ST = 'rifts2_json'
LR_POSITION = 'left'


JSON_FILE = '0_json'
########################################### DEFINE #####################################################
