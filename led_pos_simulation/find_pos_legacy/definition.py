import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import TextBox
import cv2
import matplotlib as mpl
import tkinter as tk
from collections import OrderedDict
from dataclasses import dataclass
import pickle
import gzip
import cv2
import glob
import os
from enum import Enum, auto
import math
import platform
from scipy.spatial.transform import Rotation as Rot
import json
import matplotlib.ticker as ticker
# 구 캡의 반지름 R과 높이 h
# 단위 cm
R = 8
led_r = 0.3

num_points = 150
num_leds = 15

ANGLE_SPEC = 70
DISTANCE_SPEC = 2.5

CAM_DISTANCE = 30
CAM_ANGLE = 50
angle_degrees = CAM_ANGLE
angle_radians = np.radians(angle_degrees)
sine_value = np.sin(angle_radians)

C_UPPER_Z = CAM_DISTANCE * sine_value
C_LOWER_Z = -(CAM_DISTANCE * sine_value)

READ = 0
WRITE = 1

ERROR = 'ERROR'
SUCCESS = 'SUCCESS'

DONE = 'DONE'
NOT_SET = 'NOT_SET'


camera_matrix = [
    # cam 0
    [np.array([[712.623, 0.0, 653.448],
               [0.0, 712.623, 475.572],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)],

    # cam 1
    [np.array([[716.896, 0.0, 668.902],
               [0.0, 716.896, 460.618],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.07542], [-0.026874], [0.006662], [-0.000775]], dtype=np.float64)]
]
default_dist_coeffs = np.zeros((4, 1))