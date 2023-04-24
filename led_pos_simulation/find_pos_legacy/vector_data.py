import json
import random
import signal
from collections import OrderedDict
from dataclasses import dataclass
import copy
import matplotlib.pyplot as plt
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
from enum import Enum, auto
import warnings
from dataclasses import dataclass, field
from typing import Dict
import yaml
from bleak import BleakScanner, BleakClient
import asyncio
import struct
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import time
import datetime
from scipy.spatial.transform import Rotation as R

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


def world2cam(coord, cam_pose):
    ret_coord = transfer_point(coord, cam_pose)
    print('world2cam:', ret_coord)
    return ret_coord


def camtoworld(coord, cam_pose):
    pt = copy.deepcopy(cam_pose)
    pt['position'] = vector3(0, 0, 0)
    new_pt = transfer_point(coord, pt)
    return new_pt
