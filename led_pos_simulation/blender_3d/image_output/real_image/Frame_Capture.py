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
from Advanced_Calibration import *
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
# MAX_FRAME_CNT = 360 / ANGLE
MAX_FRAME_CNT = 70
script_dir = os.path.dirname(os.path.realpath(__file__))
print(script_dir)

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(f"{script_dir}../../../../connection"))))
from connection.socket.socket_def import *




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

def init_camera_path(script_dir, video_path):
    bboxes = []
    centers1 = []
    json_file = os.path.join(script_dir, './init_blob_area.json')
    json_data = rw_json_data(READ, json_file, None)
    if json_data != ERROR:
        bboxes = json_data['bboxes']
    CAPTURE_DONE = NOT_SET
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    while(cap.isOpened()):
                # Read frames
                ret, frame = cap.read()

                if ret:
                    draw_frame = frame.copy()        
                    _, frame = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                            cv2.THRESH_TOZERO)
                            

                    height, width = frame.shape
                    center_x, center_y = width // 2, height // 2
                    cv2.line(draw_frame, (0, center_y), (width, center_y), (255, 255, 255), 1)
                    cv2.line(draw_frame, (center_x, 0), (center_x, height), (255, 255, 255), 1)       
                
                    blob_area = detect_led_lights(frame, 5, 5, 500)
                    cv2.namedWindow('image')
                    partial_click_event = functools.partial(click_event, frame=frame, blob_area_0=blob_area, bboxes=bboxes)
                    cv2.setMouseCallback('image', partial_click_event)
                    key = cv2.waitKey(1)

                    if key == ord('c'):
                        print('clear area')
                        bboxes.clear()
                    elif key == ord('s'):
                        print('save blob area')
                        json_data = OrderedDict()
                        json_data['bboxes'] = bboxes
                        # Write json data
                        rw_json_data(WRITE, json_file, json_data)
                    elif key & 0xFF == 27:
                        print('ESC pressed')
                        cv2.destroyAllWindows()
                        return
                    elif key == ord('q'):
                        print('go next step')
                        break
                    elif key == ord('n'):
                        bboxes.clear()
                        socket_cmd_to_robot('joint', 'rc', {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': ANGLE})
                        print('go next IMAGE')
                    elif key == ord('b'):
                        bboxes.clear()
                        print('go back IMAGE')
                    elif key == ord('p'):
                        CAPTURE_DONE = DONE

                    if CAPTURE_DONE == DONE:
                        print('calculation data')
                        print('bboxes', bboxes)

                        LED_NUMBERS = []
                        points2D_D = []
                        points2D_U = []
                        points3D = []
                        for AREA_DATA in bboxes:
                            IDX = int(AREA_DATA['idx'])
                            bbox = AREA_DATA['bbox']
                            (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                            gcx, gcy, gsize = find_center(frame, (x, y, w, h))
                            if gsize < BLOB_SIZE:
                                continue
                            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
                            centers1.append((gcx, gcy, bbox))
                            print(f"{IDX} : {gcx}, {gcy}")
                            LED_NUMBERS.append(IDX)
                            points3D.append(MODEL_DATA[IDX])
                            temp_blobs = np.array([gcx, gcy], dtype=np.float64)
                            points2D_D.append(temp_blobs)
                            points2D_U.append(np.array(cv2.undistortPoints(temp_blobs, camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][1])).reshape(-1, 2))
                        
            
                        print('START Pose Estimation')
                        points2D_D = np.array(np.array(points2D_D).reshape(len(points2D_D), -1), dtype=np.float64)
                        points2D_U = np.array(np.array(points2D_U).reshape(len(points2D_U), -1), dtype=np.float64)
                        points3D = np.array(points3D, dtype=np.float64)

                        # print('LED_NUMBERS: ', LED_NUMBERS)
                        # print('points2D_D\n', points2D_D)
                        # print('points2D_U\n', points2D_U)
                        # print('points3D\n', points3D)                 
                        LENGTH = len(LED_NUMBERS)
                        if LENGTH >= 4:
                            print('PnP Solver OpenCV')
                            if LENGTH >= 5:
                                METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC
                            elif LENGTH == 4:
                                METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P
                            INPUT_ARRAY = [
                                CAM_ID,
                                points3D,
                                points2D_D if undistort == 0 else points2D_U,
                                camera_matrix[CAM_ID][0] if undistort == 0 else default_cameraK,
                                camera_matrix[CAM_ID][1] if undistort == 0 else default_dist_coeffs
                            ]
                            _, rvec, tvec, _ = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
                            print('PnP_Solver rvec:', rvec.flatten(), ' tvec:',  tvec.flatten())
                            cv2.putText(draw_frame, f"rvec:{rvec.flatten()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            cv2.putText(draw_frame, f"tvec:{tvec.flatten()}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  
                        
                        INIT_IMAGE = copy.deepcopy(CAMERA_INFO_STRUCTURE)
                        INIT_IMAGE['points2D']['greysum'] = points2D_D
                        INIT_IMAGE['points2D_U']['greysum'] = points2D_U            
                        INIT_IMAGE['LED_NUMBER'] =LED_NUMBERS
                        INIT_IMAGE['points3D'] =points3D

                    draw_blobs_and_ids(draw_frame, blob_area, bboxes)
                    cv2.imshow('image', draw_frame)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    cam_dev_list = terminal_cmd('v4l2-ctl', '--list-devices')
    camera_devices = init_model_json(cam_dev_list)
    print(camera_devices)
    camera_port = camera_devices[0]['port']
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Add the directory containing poselib to the module search path
    print(script_dir)
    MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/rifts_right_9.json"))
       
    test_set = Setting_CMD()
    test_set.mv_sp = 200
    test_set.ar_coord = 'rc'
    test_set.trans_setting()
    send_cmd_to_server(sys_set)
    socket_cmd_to_robot('joint', 'ac', {'1': 1.64, '2': 71.18, '3': 19.07, '4': 0.30, '5': -89.85, '6': -197.28})
   
    init_camera_path(script_dir, camera_port)

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
