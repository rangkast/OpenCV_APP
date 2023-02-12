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


clibs = 'clibs'
sys.path.insert(0, clibs)

ENABLE = 1
DISABLE = 0

DONE = 'DONE'
NOT_SET = 'NOT_SET'

READ = 0
WRITE = 1

ERROR = 'ERROR'
SUCCESS = 'SUCCESS'
CONTINUE = 'CONTINUE'

LEFT = 0
RIGHT = 1

START = 1
STOP = 0

DEGREE_0 = 0
DEGREE_90 = 90
DEGREE_180 = 180
DEGREE_270 = 270

DISTORTION = 0
UNDISTORTION = 1

# Default
CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960

CV_MIN_THRESHOLD = 100
CV_MAX_THRESHOLD = 255

CAM_DELAY = 1

D3DR = 4
D3D = 3
D2D = 2


class POSE_ESTIMATION_METHOD(Enum):
    # Opencv legacy
    SOLVE_PNP_RANSAC = auto()
    SOLVE_PNP_REFINE_LM = auto()
    SOLVE_PNP_AP3P = auto()
    SOLVE_PNP = auto()
    SOLVE_PNP_RESERVED = auto()
    # Expansion Algorithm
    SOLVE_PNP_RECOVER_POSE = auto()
    STEREO_CAMERA = auto()


class MEAN_METHOD(Enum):
    MEAN_IQR = auto()
    MEAN_NORMAL = auto()
    MEAN_RESERVED = auto()


class DEBUG_LEVEL:
    DISABLE = -1
    LV_0 = 0
    LV_1 = 1
    LV_2 = 2
    LV_3 = 3


class LR_POSITION:
    LEFT = auto()
    RIGHT = auto()


class MODE:
    CALIBRATION_MODE = auto()
    SYSTEM_SETTING_MODE = auto()


class CAMERA_MODE:
    DUAL_CAMERA = auto()
    MULTI_CAMERA = auto()


DEBUG = DEBUG_LEVEL.LV_1

L_CONTROLLER = 'L_CONTROLLER'
R_CONTROLLER = 'R_CONTROLLER'

SENSOR_NAME_DROID = "Droid"
cam_json_droid = ['4AQ5X.json', '4AQ5P.json']

SENSOR_NAME_RIFT = "Rift"
cam_json_rift = ['WMTD307H601AF2.json', 'WMTD3064400ESF.json', ]

ORIGIN = 'rifts2_left'

# Dictionary Keys
CAM_INFO = 'cam_info'
LED_INFO = 'led_info'
MEASUREMENT_INFO = 'measurement_info'
GROUP_DATA_INFO = 'group_data_info'
SYSTEM_SETTING = 'system_setting'
BLOB_AREA_FILE = 'blob_area.json'

SYSTEM_SETTING_MODE = 'System Setting Mode'
CALIBRATION_MODE = 'Calibration Mode'
CAMERA_SETTING_MODE = 'Camera Setting Mode'

default_cameraK = np.eye(3).astype(np.float64)
default_distCoeff = np.zeros((4, 1)).astype(np.float64)

# Main Data Dictionary
ROBOT_SYSTEM_DATA = {}

# BT Config
TIMEOUT_SECONDS = 50
DEVICE_LOOK = 'RS'
BASE_SERVICE_UUID = "00001523-1212-efde-1523-785feabcd123"
CAL_TX_CHAR_UUID = "00001528-1212-efde-1523-785feabcd123"
CAL_RX_CHAR_UUID = "00001529-1212-efde-1523-785feabcd123"






