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
import cv2
import glob
import os
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

camera_log_path = "./tmp/render/camera_log.txt"
camera_img_path = "./tmp/render/"
CV_MIN_THRESHOLD = 150
CV_MAX_THRESHOLD = 255
BLOB_SIZE = 45
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

def gathering_data_single(script_dir):
    print('gathering_data_single START')
    image_files = sorted(glob.glob(os.path.join(script_dir, camera_img_path + '*.png')))
    frame_cnt = 0

    start_frame_cnt = 20
    img1 = cv2.imread(image_files[start_frame_cnt])
    _, img1 = cv2.threshold(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                cv2.THRESH_TOZERO)
    # LED blob 찾기
    blobs1 = detect_led_lights(img1, 2, 5, 500)
    # 첫 번째 이미지의 LED blob 중심점 계산
    centers1 = []
    for blob in blobs1:
        gcx, gcy, gsize = find_center(img1, blob)
        if gsize < BLOB_SIZE:
            continue
        centers1.append((gcx, gcy, blob))

    # 가장 가까운 이미지와 그 거리를 저장할 변수 초기화
    closest_img = None
    min_distance = float('inf')


    print('lenght of images: ', len(image_files))
    while frame_cnt < len(image_files) - 1:
        print('\n')
        print(f"########## Frame {frame_cnt} ##########")

        frame_0 = cv2.imread(image_files[frame_cnt])
        if frame_0 is None or frame_0.size == 0:
            print(f"Failed to load {image_files[frame_cnt]}, frame_cnt:{frame_cnt}")
            continue
        draw_frame = frame_0.copy()
        _, frame_0 = cv2.threshold(cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                                   cv2.THRESH_TOZERO)

        filename = os.path.basename(image_files[frame_cnt])
        cv2.putText(draw_frame, f"frame_cnt {frame_cnt} [{filename}]", (draw_frame.shape[1] - 300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        height, width = frame_0.shape
        center_x, center_y = width // 2, height // 2
        cv2.line(draw_frame, (0, center_y), (width, center_y), (255, 255, 255), 1)
        cv2.line(draw_frame, (center_x, 0), (center_x, height), (255, 255, 255), 1)        

        # LED blob 찾기
        blobs2 = detect_led_lights(frame_0, 2, 5, 500)
        # 두 번째 이미지의 LED blob 중심점 계산
        centers2 = []
        for blob_id, blob in enumerate(blobs2):
            gcx, gcy, gsize = find_center(frame_0, blob)
            if gsize < BLOB_SIZE:
                continue
            cv2.rectangle(draw_frame, (blob[0], blob[1]), (blob[0] + blob[2], blob[1] + blob[3]), (255, 255, 255), 1)
            cv2.putText(draw_frame, f"{blob_id}", (blob[0], blob[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            centers2.append((gcx, gcy, blob))


        for center in centers1:
            cv2.circle(draw_frame, (int(center[0]), int(center[1])), 1, (0, 255, 0), -1)

        # 두 번째 이미지의 중심점들을 이미지에 표시 (빨간색)
        for center in centers2:
            cv2.circle(draw_frame, (int(center[0]), int(center[1])), 1, (0, 0, 255), -1)

        # 두 이미지의 LED 중심점간 거리 계산
        if len(centers1) == len(centers2):
            max_distance = max(np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
                            for c1, c2 in zip(centers1, centers2))            
            # 거리가 더 작으면 업데이트
            if max_distance < min_distance and frame_cnt != start_frame_cnt:
                min_distance = max_distance
                closest_img = frame_cnt
                    
        cv2.imshow("Tracking", draw_frame)
        key = cv2.waitKey(8)
        # Exit if ESC key is
        if key & 0xFF == ord('q'):
            break

        frame_cnt += 1
    print(f'The closest image to the start image is Image_{closest_img}.')
    cv2.destroyAllWindows()

def make_video(camera_devices):
    # Select the first camera device
    camera_port = camera_devices[0]['port']

    # Open the video capture
    cap = cv2.VideoCapture(camera_port)

    # Set the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
    out = cv2.VideoWriter('output.mkv', fourcc, 60.0, (1280, 960))

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use 'mp4v' instead of 'X264'
    # out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1280, 960))
    recording = False
    if cap.isOpened():
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            draw_frame = frame.copy()
            if not ret:
                print("Unable to capture video")
                break

            # Convert the frame to grayscale
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            center_x, center_y = 1280 // 2, 960 // 2
            cv2.line(draw_frame, (0, center_y), (1280, center_y), (255, 255, 255), 1)
            cv2.line(draw_frame, (center_x, 0), (center_x, 960), (255, 255, 255), 1)     

            # Start/Stop recording
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                if not recording:
                    print('Start Recording')                
                    recording = True
            elif key & 0xFF == ord('q'):
                if recording:
                    print('Stop Recording')
                    out.release()
                    recording = False
            elif key & 0xFF == ord('e'): 
                # Use 'e' key to exit the loop
                break

            # Write the frame to file if recording
            if recording and out is not None:
                print('writing...')
                out.write(frame)
            

            # Display the resulting frame
            cv2.imshow('Frame', draw_frame)



    if out is not None:
        out.release()
    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(os.getcwd())



    cam_dev_list = terminal_cmd('v4l2-ctl', '--list-devices')
    camera_devices = init_model_json(cam_dev_list)
    print(camera_devices)


    
    make_video(camera_devices)

    # gathering_data_single(script_dir)
