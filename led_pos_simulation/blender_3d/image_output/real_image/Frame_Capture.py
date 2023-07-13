import numpy as np
import random
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import TextBox
from collections import OrderedDict
from dataclasses import dataclass
import pickle
import gzip
import os
import cv2
import glob
import matplotlib as mpl
import tkinter as tk
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import json
import matplotlib.ticker as ticker
from enum import Enum, auto
import copy
import re
import subprocess
import cv2
import traceback
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
from typing import List, Dict
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import functools
import sys
import bisect
import pprint
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from data_class import *
from itertools import combinations, permutations
import time
from threading import Lock
import threading
from queue import Queue
READ = 0
WRITE = 1
SUCCESS = 0
ERROR = -1
DONE = 1
NOT_SET = -1
CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
CV_MIN_THRESHOLD = 150
CV_MAX_THRESHOLD = 255
ANGLE = 3
MAX_FRAME_CNT = 360 / ANGLE

script_dir = os.path.dirname(os.path.realpath(__file__))
print(script_dir)

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(f"{script_dir}../../../../connection"))))
from connection.socket.socket_def import *


def detect_led_lights(image, padding=5, min_area=100, max_area=1000):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blob_info = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # Apply padding for the bounding box
        x -= padding
        y -= padding
        w += padding * 2
        h += padding * 2

        # Use contourArea to get the actual area of the contour
        area = cv2.contourArea(contour)

        # Check if the area of the contour is within the specified range
        if min_area <= area <= max_area:
            blob_info.append((x, y, w, h))

    return blob_info

def find_center(frame, SPEC_AREA):
    x_sum = 0
    t_sum = 0
    y_sum = 0
    g_c_x = 0
    g_c_y = 0
    m_count = 0

    (X, Y, W, H) = SPEC_AREA

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
        return 0, 0, 0

    return g_c_x, g_c_y, m_count


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



def init_model_json(cam_dev_list):
    print('start ', init_model_json.__name__)
    camera_info_array = []
    try:
        for i in range(len(cam_dev_list)):
            cam_info = cam_dev_list[i].split('\n\t')
            camera_info_array.append({'name': cam_info[0], 'port': cam_info[1]})
    except:
        print('exception')
        traceback.print_exc()
    finally:
        print('done')
    return camera_info_array



class WebcamStream:
    def __init__(self, src=0, status_queue=None, stop_event=None):  # Add stop_event to the constructor
        self.stream = cv2.VideoCapture(src)
        self.status_queue = status_queue
        self.frame = None
        self.frame_cnt = 0
        self.stopped = False
        self.stop_event = stop_event  # Save the stop_event
        self.file_name = 'NONE'
        # Set the resolution
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    def update(self):
        while True:
            if self.stopped:
                return
            ret, frame = self.stream.read()
            if ret:
                self.frame = frame
                if self.status_queue is not None:
                    if not self.status_queue.empty():
                        status = self.status_queue.get()
                        if status == "DONE":
                            self.file_name = f"{script_dir}/render_img/frame_{self.frame_cnt:04}.png"  # Use :04 to pad zeros up to 4 digits
                            cv2.imwrite(self.file_name, self.frame)
                            print('SAVE_IMG: ', self.file_name)
                            self.frame_cnt += 1
                            if self.frame_cnt >= MAX_FRAME_CNT:  # If the frame count is over the limit
                                print('STOP capture frame')
                                self.stream.release()
                                self.stop_event.set()  # Set the stop event to stop all threads
                                                        
                                return  # Stop the update loop
                        self.status_queue.put("NOT_SET")
                        
    def get_info(self):
        return self.file_name, self.frame_cnt

    def start(self):
        threading.Thread(target=self.update, args=()).start()
    
    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.status_queue.put("STOP")  # Add this line to signal the command_task to stop


def command_task(status_queue, stop_event):
    while True:
        if stop_event.is_set():
            break
        status = status_queue.get()
        if status == "NOT_SET":
            print('SEND_CMD')
            socket_cmd_to_robot('joint', 'rc', {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': ANGLE})
            status_queue.put("DONE")
            time.sleep(0.5)  # Wait for a while after putting "DONE" to the queue

if __name__ == "__main__":
    cam_dev_list = terminal_cmd('v4l2-ctl', '--list-devices')
    camera_devices = init_model_json(cam_dev_list)
    print(camera_devices)
    camera_port = camera_devices[0]['port']
    
    test_set = Setting_CMD()
    test_set.mv_sp = 200
    test_set.ar_coord = 'rc'
    test_set.trans_setting()
    send_cmd_to_server(sys_set)
    socket_cmd_to_robot('joint', 'ac', {'1': 1.64, '2': 71.18, '3': 19.07, '4': 0.30, '5': -89.85, '6': -197.28})
   
    status_queue = Queue()
    status_queue.put("NOT_SET")
    stop_event = threading.Event()  # Create a stop event
    webcam_stream = WebcamStream(src=camera_port, status_queue=status_queue, stop_event=stop_event)  # Pass the stop event to the WebcamStream
    webcam_stream.start()
    threading.Thread(target=command_task, args=(status_queue, stop_event)).start()
    
    while True:
        frame = webcam_stream.read()
        file_name, frame_cnt = webcam_stream.get_info()
        if frame is not None:
            draw_frame = frame.copy()
            center_x, center_y = CAP_PROP_FRAME_WIDTH // 2, CAP_PROP_FRAME_HEIGHT // 2
            cv2.line(draw_frame, (0, center_y), (CAP_PROP_FRAME_WIDTH, center_y), (255, 255, 255), 1)
            cv2.line(draw_frame, (center_x, 0), (center_x, CAP_PROP_FRAME_HEIGHT), (255, 255, 255), 1)
            cv2.putText(draw_frame, f"frame_cnt {frame_cnt} [{file_name}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)  
            
            # Display the resulting frame
            cv2.imshow('Frame', draw_frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or stop_event.is_set():
                break

    webcam_stream.stop()
    stop_event.set()
    cv2.destroyAllWindows()
