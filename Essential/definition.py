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


def init_model_json(path, cam_dev_list):
    print('start ', init_model_json.__name__)
    camera_info_array = []
    try:
        # if DEBUG == ENABLE:
        #     print(cam_dev_list, ' count:', len(cam_dev_list))
        for i in range(len(cam_dev_list)):
            # print('\n')
            # print('cam_id[', i, '] :', cam_json[i])
            jsonObject = json.load(open(''.join([f'{path}', '/jsons/', f'{cam_json[i]}'])))

            # cam_info = cam_dev_list[i].split('\n\t')

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

            # if DEBUG == ENABLE:
            #     print('cameraK: ', A)
            #     print('dist_coeff: ', K)

            camera_info_array.append({'idx': i,
                                      # 'name': cam_info[0],
                                      # 'port': cam_info[1],
                                      'json': cam_json[i],
                                      'cam_cal': {'cameraK': A, 'dist_coeff': K},

                                      'blobs': [],
                                      'med_blobs': [],

                                      'distorted_2d': [],
                                      'undistorted_2d': [],

                                      'track_cal': {'data': [], 'recording': {'name': NOT_SET}},

                                      'D_R_T': {'rvecs': NOT_SET, 'tvecs': NOT_SET},
                                      'D_R_T_A': [],
                                      'RER': {'C_R_T': {'rvecs': NOT_SET, 'tvecs': NOT_SET}}})

    except:
        print('exception')
        traceback.print_exc()
    finally:
        print('done')
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
                             'remake_3d': []}

            print(''.join(['{ .pos = {{', f'{x}', ',', f'{y}', ',', f'{z}',
                           ' }}, .dir={{', f'{u}', ',', f'{v}', ',', f'{w}', ' }}, .pattern=', f'{idx}', '},']))
    except:
        print('exception')
        traceback.print_exc()
    finally:
        print('done')
    return pts


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


def rw_file_storage(rw_cmd, left_map, right_map):
    if rw_cmd == WRITE:
        print("WRITE parameters ......")
        cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_WRITE)
        cv_file.write("Left_Stereo_Map_x", left_map[0])
        cv_file.write("Left_Stereo_Map_y", left_map[1])
        cv_file.write("Right_Stereo_Map_x", right_map[0])
        cv_file.write("Right_Stereo_Map_y", right_map[1])
        cv_file.release()
    else:
        print("READ parameters ......")
        try:
            # FILE_STORAGE_READ
            cv_file = cv2.FileStorage("improved_params2.xml", cv2.FILE_STORAGE_READ)
            # note we also have to specify the type to retrieve other wise we only get a
            # FileNode object back instead of a matrix
            left_map = (cv_file.getNode("Left_Stereo_Map_x").mat(), cv_file.getNode("Left_Stereo_Map_y").mat())
            right_map = (cv_file.getNode("Right_Stereo_Map_x").mat(), cv_file.getNode("Right_Stereo_Map_y").mat())

            cv_file.release()

            return DONE, left_map, right_map
        except:
            traceback.print_exc()
            return ERROR, NOT_SET, NOT_SET

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
        pts.append({'idx': idx, 'pos': _pos, 'dir': _dir, 'pair_xy': [], 'remake_3d': [],
                    'min': {'dis': 10, 'remake': []}})

    # print(f'{fname} PointsRead')
    leds_dic[fname] = pts


def zoom_factory(ax, base_scale=2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
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
        plt.draw()  # force re-draw

    fig = ax.get_figure()  # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event', zoom_fun)
    # return the function
    return zoom_fun


# add common function here
def draw_dots(dimension, pts, ax, c):
    if dimension == 3:
        x = [x['pos'][0] for x in pts]
        y = [x['pos'][1] for x in pts]
        z = [x['pos'][2] for x in pts]
        u = [x['dir'][0] for x in pts]
        v = [x['dir'][1] for x in pts]
        w = [x['dir'][2] for x in pts]
        idx = [x['idx'] for x in pts]
        ax.scatter(x, y, z, marker='o', s=5, color=c, alpha=0.5)
        ax.quiver(x, y, z, u, v, w, length=0.05, linewidths=0.1, color='red', normalize=True)
        for idx, x, y, z in zip(idx, x, y, z):
            label = '%s' % idx
            ax.text(x, y, z, label, size=5)

    elif dimension == 2:
        x = [x['pos'][0] for x in pts]
        y = [x['pos'][1] for x in pts]
        idx = [x['idx'] for x in pts]
        ax.scatter(x, y, marker='o', s=5, color=c, alpha=0.5)
        # 좌표 numbering
        if c == 'red':
            for idx, x, y in zip(idx, x, y):
                label = '%s' % idx
                ax.text(x + 0.001, y + 0.001, label, size=7)




leds_dic = {}

ENABLE = 1
DISABLE = 0

D3DT = 4
D3D = 3
D2D = 2

READ = 0
WRITE = 1

ERROR = -1
SUCCESS = 1

DONE = 'DONE'
NOT_SET = 'NOT_SET'

SUDO_PASSWORD = ''

# CAMERA
cameraK = np.eye(3).astype(np.float64)
distCoeff = np.zeros((4, 1)).astype(np.float64)

DEBUG = ENABLE

LOOP_CNT = 100

CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960

MAX_BLOB_SIZE = 250

CV_FINDCONTOUR_LVL = 140
CV_THRESHOLD = 170
CV_MIN_THRESHOLD = 170
CV_MID_THRESHOLD = 190
CV_MAX_THRESHOLD = 255

DO_ESTIMATE_POSE = ENABLE
DO_SOLVEPNP_REFINE = DISABLE
DO_UNDISTORT = DISABLE
DO_SOLVEPNP_RANSAC = ENABLE
USE_PRINT_FRAME = DISABLE

LED_COUNT = 15

ORIGIN = 'rifts2_left.json'
JSON_FILE = 'stereo_json'

cam_json = ['WMTD307H601E9L.json', 'WMTD302J600GA9.json']
