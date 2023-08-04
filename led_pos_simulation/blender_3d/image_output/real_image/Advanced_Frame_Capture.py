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
CV_MIN_THRESHOLD = 100
CV_MAX_THRESHOLD = 255
ANGLE = 3
MAX_FRAME_CNT = 360 / ANGLE
# MAX_FRAME_CNT = 70
BLOB_SIZE = 20
TRACKER_PADDING = 3
DO_SOCKET_COMM = 1

UP = 1
DOWN = -1
MOVE_DIRECTION = UP
DIRECTION_CNT = 10

script_dir = os.path.dirname(os.path.realpath(__file__))
print(script_dir)

PREV_BLOB = {}
NEW_CAMERA_INFO = {}
NEW_CAMERA_INFO_STRUCTURE = {
    'ANGLE': NOT_SET,
    'LED_NUMBER': [],
    'points2D': [],
}

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(f"{script_dir}../../../../connection"))))
from connection.socket.socket_def import *

def init_trackers(trackers, frame):
    for id, data in trackers.items():
        tracker = cv2.TrackerCSRT_create()
        (x, y, w, h) = data['bbox']        
        ok = tracker.init(frame, (x - TRACKER_PADDING, y - TRACKER_PADDING, w + 2 * TRACKER_PADDING, h + 2 * TRACKER_PADDING))
        data['tracker'] = tracker

def clear_tracker(trackers):
    for id, data in trackers.items():
        print(data)
        tracker = data['tracker']
        tracker.clear

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
        self.frame_cnt = -1
        self.vertical_cnt = -1
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
                        if status[0] == "DONE":
                            # self.frame_cnt += 1
                            self.frame_cnt = status[1]
                            self.vertical_cnt = status[2]
                            print(f"!!! frame_cnt {self.frame_cnt} vertical_cnt {self.vertical_cnt} !!!")
                            if self.vertical_cnt >= DIRECTION_CNT or self.vertical_cnt == 5:                                
                                # Test Code
                                LED_NUMBER = []
                                points2D = []
                                if self.frame_cnt >= 0 and PREV_BLOB.get(self.frame_cnt) != None:
                                    for i, blob_data in enumerate(PREV_BLOB[self.frame_cnt]):
                                        LED_NUMBER.append(int(blob_data[4]))
                                        points2D.append([blob_data[0], blob_data[1]])
                         
                                        pt = (int(blob_data[0]), int(blob_data[1]))
                                        # cv2.circle(self.frame, pt, 2, (255, 0, 0), -1)
                                        if blob_data[4] != -1:
                                            cv2.putText(self.frame, f"{blob_data[4]}: [{int(blob_data[0])},{int(blob_data[1])}]", (CAP_PROP_FRAME_WIDTH - 200, 50 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255, 0, 0), 1)
                                KEY = f"{self.frame_cnt:04}_{MOVE_DIRECTION}_{self.vertical_cnt}"
                                NEW_CAMERA_INFO[KEY] = copy.deepcopy(NEW_CAMERA_INFO_STRUCTURE)
                                points2D = np.array(np.array(points2D).reshape(len(points2D), -1), dtype=np.float64)                                

                                NEW_CAMERA_INFO[KEY]['points2D'] = points2D
                                NEW_CAMERA_INFO[KEY]['ANGLE'] = -MOVE_DIRECTION * ANGLE * self.vertical_cnt
                                NEW_CAMERA_INFO[KEY]['LED_NUMBER'] =LED_NUMBER
                               
                                self.file_name = f"{script_dir}/render_img/capture/frame_{self.frame_cnt:04}_{MOVE_DIRECTION}_{self.vertical_cnt}.png"  # Use :04 to pad zeros up to 4 digits
                                cv2.imwrite(self.file_name, self.frame)
 
                                # print('NEW_CAMERA_INFO')
                                # print(NEW_CAMERA_INFO)
            
                                print('SAVE_IMG: ', self.file_name)

                            else:
                                print(f"tracking move {self.vertical_cnt}")
                 
                            if self.frame_cnt >= MAX_FRAME_CNT:  # If the frame count is over the limit
                                print('STOP capture frame')
                                self.stream.release()
                                self.stop_event.set()  # Set the stop event to stop all threads                                                        
                                return  # Stop the update loop
                        self.status_queue.put(["NOT_SET", self.frame_cnt, self.vertical_cnt])
                        
    def get_info(self):
        return self.file_name, self.frame_cnt, self.vertical_cnt

    def start(self):
        threading.Thread(target=self.update, args=()).start()
    
    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.status_queue.put(["STOP", -1])  # Add this line to signal the command_task to stop


def command_task(status_queue, stop_event):
    HOPPING_CNT = 5
    vertical_cnt = 0
    frame_cnt = 0
    while True:
        if stop_event.is_set():
            break
        status = status_queue.get()
        if status[0] == "NOT_SET":
            # print('SEND_CMD')           
            # START POINTS
            if status[1] == -1:
                socket_cmd_to_robot('joint', 'rc', {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': ANGLE})
            else:
                if vertical_cnt >= DIRECTION_CNT:
                    if MOVE_DIRECTION == UP:
                        if DO_SOCKET_COMM == DONE:
                            socket_cmd_to_robot('joint', 'rc', {'1': 0, '2': 0, '3': 0, '4': 0, '5': ANGLE * DOWN * vertical_cnt, '6': 0})
                    elif MOVE_DIRECTION == DOWN:
                        if DO_SOCKET_COMM == DONE:
                            socket_cmd_to_robot('joint', 'rc', {'1': 0, '2': 0, '3': 0, '4': 0, '5': ANGLE * UP * vertical_cnt, '6': 0})
                    if DO_SOCKET_COMM == DONE:
                        socket_cmd_to_robot('joint', 'rc', {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': ANGLE})
                    vertical_cnt = 0
                    frame_cnt += 1
                else:
                    if MOVE_DIRECTION == UP:
                        if DO_SOCKET_COMM == DONE:
                            socket_cmd_to_robot('joint', 'rc', {'1': 0, '2': 0, '3': 0, '4': 0, '5': ANGLE * UP * HOPPING_CNT, '6': 0})
                    elif MOVE_DIRECTION == DOWN:
                        if DO_SOCKET_COMM == DONE:
                            socket_cmd_to_robot('joint', 'rc', {'1': 0, '2': 0, '3': 0, '4': 0, '5': ANGLE * DOWN * HOPPING_CNT, '6': 0})     
                    vertical_cnt += HOPPING_CNT                   

            # print('sendcmd ', frame_cnt, ' ', vertical_cnt)
            status_queue.put(["DONE", frame_cnt, vertical_cnt])
            time.sleep(0.3)  # Wait for a while after putting "DONE" to the queue

def init_camera_path(script_dir, video_path):
    bboxes = []
    centers1 = []
    TEMP_POS = {'status': NOT_SET, 'mode': RECTANGLE, 'start': [], 'move': [], 'circle': NOT_SET, 'rectangle': NOT_SET}
    POS = {}
    json_file = os.path.join(script_dir, './init_blob_area.json')
    json_data = rw_json_data(READ, json_file, None)
    if json_data != ERROR:
        bboxes = json_data['bboxes']
        if 'mode' in json_data:
            TEMP_POS['mode'] = json_data['mode']
            if TEMP_POS['mode'] == RECTANGLE:
                TEMP_POS['rectangle'] = json_data['roi']
            else:
                TEMP_POS['circle'] = json_data['roi']        
            POS = TEMP_POS
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
                    filtered_blob_area = []    
                    for _, bbox in enumerate(blob_area):
                        (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                        gcx,gcy, gsize = find_center(frame, (x, y, w, h))
                        if gsize < BLOB_SIZE:
                            continue
                        filtered_blob_area.append((gcx, gcy, (x, y, w, h)))            
                    cv2.namedWindow('image')
                    partial_click_event = functools.partial(click_event, frame=frame, blob_area_0=filtered_blob_area, bboxes=bboxes, POS=TEMP_POS)
                    cv2.setMouseCallback('image', partial_click_event)

                            
                    if TEMP_POS['status'] == MOVE:
                        dx = np.abs(TEMP_POS['start'][0] - TEMP_POS['move'][0])
                        dy = np.abs(TEMP_POS['start'][1] - TEMP_POS['move'][1])
                        if TEMP_POS['mode'] == CIRCLE:
                            radius = math.sqrt(dx ** 2 + dy ** 2) / 2
                            cx = int((TEMP_POS['start'][0] + TEMP_POS['move'][0]) / 2)
                            cy = int((TEMP_POS['start'][1] + TEMP_POS['move'][1]) / 2)
                            # print(f"{dx} {dy} radius {radius}")
                            TEMP_POS['circle'] = [cx, cy, radius]
                        else:                
                            TEMP_POS['rectangle'] = [TEMP_POS['start'][0],TEMP_POS['start'][1],TEMP_POS['move'][0],TEMP_POS['move'][1]]
                
                    elif TEMP_POS['status'] == UP:
                        print(TEMP_POS)
                        TEMP_POS['status'] = NOT_SET        
                    if TEMP_POS['circle'] != NOT_SET and TEMP_POS['mode'] == CIRCLE:
                        cv2.circle(draw_frame, (TEMP_POS['circle'][0], TEMP_POS['circle'][1]), int(TEMP_POS['circle'][2]), (255,255,255), 1)
                    elif TEMP_POS['rectangle'] != NOT_SET and TEMP_POS['mode'] == RECTANGLE:
                        cv2.rectangle(draw_frame,(TEMP_POS['rectangle'][0],TEMP_POS['rectangle'][1]),(TEMP_POS['rectangle'][2],TEMP_POS['rectangle'][3]),(255,255,255),1)


                    key = cv2.waitKey(1)

                    if key == ord('c'):
                        print('clear area')
                        bboxes.clear()
                        if TEMP_POS['mode'] == RECTANGLE:
                            TEMP_POS['rectangle'] = NOT_SET
                        else:
                            TEMP_POS['circle'] = NOT_SET
                    elif key == ord('s'):
                        print('save blob area')
                        json_data = OrderedDict()
                        json_data['bboxes'] = bboxes
                        json_data['mode'] = TEMP_POS['mode']
                        json_data['roi'] = TEMP_POS['circle'] if TEMP_POS['mode'] == CIRCLE else TEMP_POS['rectangle']
                        POS = TEMP_POS
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

                    draw_blobs_and_ids(draw_frame, filtered_blob_area, bboxes)
                    cv2.imshow('image', draw_frame)

    cv2.destroyAllWindows()
def tracker_operation(frame, draw_frame, frame_cnt):
    # Tracker Operation
    # INIT Tracker
    # 다음 프레임에 갈 때마다 초기화
    if prev_frame_cnt != frame_cnt:
        print(f"############### {frame_cnt} ####################")
        # print('CAMERA_INFO')
        print(CAMERA_INFO[f"{frame_cnt}"])
        
        # ToDo 
        TRACKING_START = NOT_SET
        if TRACKING_START == NOT_SET:                    
            print('REINIT Tracker')
            # clear_tracker(CURR_TRACKER)
            bboxes = []
            CURR_TRACKER.clear()     
            PREV_TRACKER.clear()       
            bboxes = CAMERA_INFO[f"{frame_cnt}"]['bboxes']
            print('bboxes:', bboxes)
            if bboxes is None:
                return ERROR            
            for i in range(len(bboxes)):
                CURR_TRACKER[bboxes[i]['idx']] = {'bbox': bboxes[i]['bbox'], 'tracker': None}
            init_trackers(CURR_TRACKER, frame)          
            TRACKING_START = DONE

    if TRACKING_START == DONE:
        # find Blob area by findContours
        blob_area = detect_led_lights(frame, TRACKER_PADDING, 5, 500)
        blobs = []
        
        for blob_id, bbox in enumerate(blob_area):
            (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            gcx, gcy, gsize = find_center(frame, (x, y, w, h))
            if gsize < BLOB_SIZE:
                continue
            # cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
            # cv2.putText(draw_frame, f"{blob_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            blobs.append((gcx, gcy, bbox))

        CURR_TRACKER_CPY = CURR_TRACKER.copy()
        for Tracking_ANCHOR, Tracking_DATA in CURR_TRACKER_CPY.items():
            if Tracking_DATA['tracker'] is not None:
                ret, (tx, ty, tw, th) = Tracking_DATA['tracker'].update(frame)
                cv2.rectangle(draw_frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 1, 1)
                cv2.putText(draw_frame, f"{Tracking_ANCHOR}", (tx, ty + th + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                tcx, tcy, tsize = find_center(frame, (tx, ty, tw, th))
                if Tracking_ANCHOR in PREV_TRACKER:
                    def check_distance(blob_centers, tcx, tcy):
                        for center in blob_centers:
                            gcx, gcy, _ = center
                            distance = math.sqrt((gcx - tcx)**2 + (gcy - tcy)**2)
                            if distance < 1:
                                return False
                        return True
                    dx = PREV_TRACKER[Tracking_ANCHOR][0] - tcx
                    dy = PREV_TRACKER[Tracking_ANCHOR][1] - tcy
                    euclidean_distance = math.sqrt(dx ** 2 + dy ** 2)
                    # 트랙커가 갑자기 이동
                    # 사이즈가 작은 경우
                    # 실패한 경우
                    # 중심점위치에 Blob_center 데이터가 없는 경우
                    exist_status = check_distance(blobs, tcx, tcy)
                    if exist_status or euclidean_distance > 5 or tsize < BLOB_SIZE or not ret:
                        # print('Tracker Broken')
                        # print('euclidean_distance:', euclidean_distance, ' tsize:', tsize, ' ret:', ret, 'exist_status:', exist_status)
                        # print('CUR_txy:', tcx, tcy)
                        # print('PRV_txy:', PREV_TRACKER[Tracking_ANCHOR])
                        if ret == SUCCESS:
                            del CURR_TRACKER[Tracking_ANCHOR]
                            del PREV_TRACKER[Tracking_ANCHOR]
                            # 여기서 PREV에 만들어진 위치를 집어넣어야 바로 안튕김
                            print(f"tracker[{Tracking_ANCHOR}] deleted")
                        #     continue
                        # else:
                        #     break

            PREV_TRACKER[Tracking_ANCHOR] = (tcx, tcy, (tx, ty, tw, th))

def distance_operation(frame, draw_frame, frame_cnt, prev_frame_cnt, vertical_cnt):


    if frame_cnt > MAX_FRAME_CNT - 1:
        print('EOF CAMERA_INFO')
        return

    # find Blob area by findContours
    blob_area = detect_led_lights(frame, TRACKER_PADDING, 5, 500)
    blobs = []

    for blob_id, bbox in enumerate(blob_area):
        (x, y, w, h) = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        gcx, gcy, gsize = find_center(frame, (x, y, w, h))
        # if gsize < BLOB_SIZE:
        #     continue
        cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (255, 255, 255), 1, 1)
        # cv2.putText(draw_frame, f"{blob_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        blobs.append([gcx, gcy, bbox, NOT_SET, -1, float('inf')])
    
    TRACKER = []
    if frame_cnt >= 0:
        bboxes = CAMERA_INFO[f"{frame_cnt}"]['bboxes']
        for tracker_data in bboxes:
            tcx, tcy, _ = find_center(frame, tracker_data['bbox'])
            if prev_frame_cnt != frame_cnt:
                TRACKER.append([tracker_data['idx'], [tcx, tcy]])
            (tx, ty, tw, th) = tracker_data['bbox']
            # cv2.rectangle(draw_frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 1, 1)
            cv2.putText(draw_frame, f"{tracker_data['idx']}", (tx, ty + th + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(draw_frame, (int(tcx), int(tcy)), 2, (0, 255, 0), -1)            


    # print(f"{prev_frame_cnt} {frame_cnt} {vertical_cnt}")
    searching_blob = copy.deepcopy(blobs)
    if prev_frame_cnt != frame_cnt:
        # print('bboxes:', bboxes)
        # TRACKER = []
        # for tracker_data in bboxes:
        #     tcx, tcy, _ = find_center(frame, tracker_data['bbox'])
        #     TRACKER.append([tracker_data['idx'], [tcx, tcy]])
        

        for t_data in TRACKER:
            blob_id = t_data[0]
            tcxy = t_data[1]

            min_distance = float('inf')
            min_idx = -1
            for i, b_data in enumerate(searching_blob):
                if blobs[i][3] == DONE:
                    continue
                dx = np.abs(b_data[0] - tcxy[0])
                dy = np.abs(b_data[1] - tcxy[1])
                distance = math.sqrt(dx ** 2 + dy ** 2)

                # distance = np.linalg.norm(np.array([b_data[0], b_data[1]]) - np.array(tcxy), axis=0)

                if distance < min_distance:
                    min_distance = distance
                    min_idx = i
            
            if min_idx != -1:
                blobs[min_idx][5] = min_distance
                blobs[min_idx][4] = blob_id
                blobs[min_idx][3] = DONE
        # print(f"###  frame_cnt ### {frame_cnt} init")
        PREV_BLOB[frame_cnt] = blobs

    else:
        if frame_cnt >= 0:
            for t_data in PREV_BLOB[frame_cnt]:
                blob_id = t_data[4]
                tcx = t_data[0]
                tcy = t_data[1]
                prev_distance = t_data[5]

                if blob_id == -1:
                    continue

                min_distance = float('inf')
                min_idx = -1
                for i, b_data in enumerate(searching_blob):
                    if blobs[i][3] == DONE:
                        continue
                    dx = np.abs(b_data[0] - tcx)
                    dy = np.abs(b_data[1] - tcy)
                    distance = math.sqrt(dx ** 2 + dy ** 2)

                    if distance < min_distance:
                        min_distance = distance
                        min_idx = i
                
                if min_idx != -1:
                    # print(f"blob_id{blob_id} {prev_distance - min_distance}")
                    dist_diff = np.abs(prev_distance - min_distance)
                    if dist_diff >= 10:
                        blobs[min_idx][4] = -1                        
                    else:
                        blobs[min_idx][4] = blob_id
                        blobs[min_idx][5] = min_distance

                    blobs[min_idx][3] = DONE
            # print(f"###  frame_cnt ### {frame_cnt} update ")
            PREV_BLOB[frame_cnt] = blobs


    # if vertical_cnt == 10:
    #     print('blobs')
    #     print(blobs)
    #     print(PREV_BLOB[frame_cnt])

    if frame_cnt >= 0:
        for blob_data in PREV_BLOB[frame_cnt]:                            
            pt = (int(blob_data[0]), int(blob_data[1]))
            cv2.circle(draw_frame, pt, 2, (0, 0, 255), -1)
            if blob_data[4] != -1:
                cv2.putText(draw_frame, str(blob_data[4]), (int(blob_data[0]), int(blob_data[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 0, 255), 1,)


if __name__ == "__main__":
    cam_dev_list = terminal_cmd('v4l2-ctl', '--list-devices')
    camera_devices = init_model_json(cam_dev_list)
    print(camera_devices)
    camera_port = camera_devices[0]['port']
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Add the directory containing poselib to the module search path
    print(script_dir)
    MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/rifts_right_9.json"))
    # READ CAMERA_INFO STRUCTURE for Labeling IDS
    CAMERA_INFO = pickle_data(READ, 'CAMERA_INFO.pickle', None)['CAMERA_INFO']           

    ############################## TOP BAR ##############################
    #
    # 430 200 30
    #
    if DO_SOCKET_COMM == DONE:
        test_set = Setting_CMD()
        test_set.mv_sp = 200
        test_set.ar_coord = 'rc'
        test_set.trans_setting()
        send_cmd_to_server(sys_set)
        socket_cmd_to_robot('joint', 'ac', {'1': 1.64, '2': 71.18, '3': 19.07, '4': 0.30, '5': -89.85, '6': -197.28})
   
    # init_camera_path(script_dir, camera_port)

    status_queue = Queue()
    status_queue.put(["NOT_SET", -1])
    stop_event = threading.Event()  # Create a stop event
    webcam_stream = WebcamStream(src=camera_port, status_queue=status_queue, stop_event=stop_event)  # Pass the stop event to the WebcamStream
    webcam_stream.start()
    threading.Thread(target=command_task, args=(status_queue, stop_event)).start()
    
    # TRACKING_START = NOT_SET
    # CURR_TRACKER = {}
    # PREV_TRACKER = {}
    # bboxes = []
    prev_frame_cnt = -1

    while True:
        frame = webcam_stream.read()
        file_name, frame_cnt, vertical_cnt = webcam_stream.get_info()

        if frame is not None:
            draw_frame = frame.copy()
            _, frame = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                  CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                  cv2.THRESH_TOZERO)
            center_x, center_y = CAP_PROP_FRAME_WIDTH // 2, CAP_PROP_FRAME_HEIGHT // 2
            cv2.line(draw_frame, (0, center_y), (CAP_PROP_FRAME_WIDTH, center_y), (255, 255, 255), 1)
            cv2.line(draw_frame, (center_x, 0), (center_x, CAP_PROP_FRAME_HEIGHT), (255, 255, 255), 1)
            cv2.putText(draw_frame, f"frame_cnt {frame_cnt} {-ANGLE * MOVE_DIRECTION * vertical_cnt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)  
            
            # tracker_operation(frame, draw_frame, frame_cnt)
            distance_operation(frame, draw_frame, frame_cnt, prev_frame_cnt, vertical_cnt)

            prev_frame_cnt = frame_cnt
            cv2.imshow('Frame', draw_frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or stop_event.is_set():                
                break
            elif key & 0xFF == ord('t'):
                print('tracker start')        
                TRACKING_START = NOT_SET
                frame_cnt += 1
            elif key & 0xFF == ord('n'):
                status_queue.put(["NOT_SET", frame_cnt, vertical_cnt])
                
    webcam_stream.stop()
    stop_event.set()
    cv2.destroyAllWindows()

    # file = f"NEW_CAMERA_INFO_{MOVE_DIRECTION}.pickle"
    # data = OrderedDict()
    # data['NEW_CAMERA_INFO'] = NEW_CAMERA_INFO
    # ret = pickle_data(WRITE, file, data)
    # if ret != ERROR:
    #     print('data saved')