import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import TextBox
import cv2
import matplotlib as mpl
import tkinter as tk

# 구 캡의 반지름 R과 높이 h
# 단위 cm
R = 11
r = 0.3

num_points = 150
num_leds = 15

ANGLE_SPEC = 70
DISTANCE_SPEC = 1.3

CAM_DISTANCE = 30
CAM_ANGLE = 60
angle_degrees = CAM_ANGLE
angle_radians = np.radians(angle_degrees)
sine_value = np.sin(angle_radians)

C_UPPER_Z = CAM_DISTANCE * sine_value
C_LOWER_Z = -(CAM_DISTANCE * sine_value)