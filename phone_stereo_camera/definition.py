from stereo_camera import *

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
import open3d as o3d
import os
import sys


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


def rw_file_storage(rw_cmd, name, left_map, right_map):
    if rw_cmd == WRITE:
        print("WRITE parameters ......")
        cv_file = cv2.FileStorage(name, cv2.FILE_STORAGE_WRITE)
        cv_file.write("Left_Stereo_Map_x", left_map[0])
        cv_file.write("Left_Stereo_Map_y", left_map[1])
        cv_file.write("Right_Stereo_Map_x", right_map[0])
        cv_file.write("Right_Stereo_Map_y", right_map[1])
        cv_file.release()
    else:
        print("READ parameters ......")
        try:
            # FILE_STORAGE_READ
            cv_file = cv2.FileStorage(name, cv2.FILE_STORAGE_READ)
            # note we also have to specify the type to retrieve other wise we only get a
            # FileNode object back instead of a matrix
            left_map = (cv_file.getNode("Left_Stereo_Map_x").mat(), cv_file.getNode("Left_Stereo_Map_y").mat())
            right_map = (cv_file.getNode("Right_Stereo_Map_x").mat(), cv_file.getNode("Right_Stereo_Map_y").mat())

            cv_file.release()

            return DONE, left_map, right_map
        except:
            traceback.print_exc()
            return ERROR, NOT_SET, NOT_SET


def Rotate(src, degrees):
    if degrees == 90:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 1)

    elif degrees == 180:
        dst = cv2.flip(src, -1)

    elif degrees == 270:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 0)
    else:
        dst = NOT_SET
    return dst


CAM_1 = ["/dev/video0", "imgL"]
CAM_2 = ["/dev/video1", "imgR"]

CAP_PROP_FRAME_WIDTH = 1920
CAP_PROP_FRAME_HEIGHT = 1080

# Defining the dimensions of checkerboard
CHECKERBOARD = (7, 4)
# Termination criteria for refining the detected corners
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

ENABLE = 1
DISABLE = 0
DONE = 'DONE'
NOT_SET = 'NOT_SET'
READ = 0
WRITE = 1
ERROR = -1
SUCCESS = 1

JSON_FILE = 'stereo_json'
EXTERNAL_TOOL_CALIBRATION = 'calibration_json'
RECTIFY_MAP = "improved_params2.xml"
CAM_DELAY = 1

USE_EXTERNAL_TOOL_CALIBRAION = DISABLE

data_info_dictionary = {'display': {'left': [CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT],
                                    'right': [CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT]}}

