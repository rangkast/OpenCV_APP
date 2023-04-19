import matplotlib.pyplot as plt
import numpy as np
import qdarkstyle

import json
import random
import signal
from collections import OrderedDict
from dataclasses import dataclass
import copy
from scipy.spatial.transform import Rotation as R
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
from PIL import Image
from os import path

CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
CV_MIN_THRESHOLD = 100
CV_MAX_THRESHOLD = 255

print('PATH', path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
PATH = '../datasets/led_controller'


def dump_data(number, image, bboxes, boxes, size):
    img = Image.fromarray(image)
    formatted_number = "{:08d}".format(number)
    file_name = "{}.txt".format(formatted_number)
    img_name = "{}.bmp".format(formatted_number)

    content_str = []
    for i, bp in enumerate(boxes):
        (x, y, w, h) = bp
        cx = int((x + w) / 2)
        cy = int((y + h) / 2)
        content_str.append(str(bboxes[i]['idx']) + " "
                           + str('%0.2f' % (cx / CAP_PROP_FRAME_WIDTH)) + " "
                           + str('%0.2f' % (cy / CAP_PROP_FRAME_HEIGHT)) + " "
                           + str('%0.2f' % (w / CAP_PROP_FRAME_WIDTH)) + " "
                           + str('%0.2f' % (h / CAP_PROP_FRAME_HEIGHT)))

    # print(content_str)
    # 파일을 쓰기 모드로 열기
    with open(''.join([PATH, '/labels/'f'{file_name}']), 'w') as file:
        # 파일에 내용 작성
        for content in content_str:
            file.write(content + '\n')

    img.save(''.join([PATH, '/images/'f'{img_name}']))


def view_camera_infos(frame, text, x, y):
    cv2.putText(frame, text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), lineType=cv2.LINE_AA)


trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.legacy.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)

    return tracker


def make_dataset_tracker(cam_id):
    bboxes = []
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FORMAT, cv2.CV_64FC1)
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Cannot read video file')
            break
        ret, img_contour_binary = cv2.threshold(frame, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
        img_gray = cv2.cvtColor(img_contour_binary, cv2.COLOR_BGR2GRAY)
        img_draw = frame.copy()

        KEY = cv2.waitKey(1)

        if KEY == ord('a'):
            cv2.imshow('MultiTracker', img_gray)
            bbox = cv2.selectROI('MultiTracker', img_gray)
            while True:
                inputs = input('input led number: ')
                if inputs.isdigit():
                    input_number = int(inputs)
                    if input_number in range(0, 25):
                        print('label number ', input_number)
                        print('bbox ', bbox)
                        bboxes.append({'idx': input_number, 'bbox': bbox})
                        break
                elif cv2.waitKey(1) == ord('q'):
                    bboxes.clear()
                    break

        elif KEY == ord('n'):
            break

        elif KEY & 0xFF == 27:
            print('ESC pressed')
            cap.release()
            cv2.destroyAllWindows()
            return

        if len(bboxes) > 0:
            for i, data in enumerate(bboxes):
                (x, y, w, h) = data['bbox']
                IDX = data['idx']
                cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
                view_camera_infos(img_draw, ''.join([f'{IDX}']), int(x), int(y) - 10)
                view_camera_infos(img_draw, ''.join(['[', f'{IDX}', '] '
                                                        , f' {x}'
                                                        , f' {y}']), 30, 35 + i * 30)

        cv2.circle(img_draw, (int(CAP_PROP_FRAME_WIDTH / 2), int(CAP_PROP_FRAME_HEIGHT / 2)), 2, color=(0, 0, 255),
                   thickness=-1)
        cv2.imshow('MultiTracker', img_draw)

    # end while
    print('Selected bounding boxes {}'.format(bboxes))

    # Specify the tracker type
    trackerType = "CSRT"

    # Create MultiTracker object
    multiTracker = cv2.legacy.MultiTracker_create()

    tracker_start = 0
    start_capture = 0
    CNT = 0
    # Process video and track objects
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, img_contour_binary = cv2.threshold(frame, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
        img_gray = cv2.cvtColor(img_contour_binary, cv2.COLOR_BGR2GRAY)

        # Initialize MultiTracker
        if tracker_start == 0:
            for i, data in enumerate(bboxes):
                multiTracker.add(createTrackerByName(trackerType), img_gray, data['bbox'])

        tracker_start = 1

        # get updated location of objects in subsequent frames
        qq, boxes = multiTracker.update(img_gray)

        img_draw = frame.copy()
        # draw tracked objects
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(img_draw, p1, p2, 255, 1)
            IDX = bboxes[i]['idx']
            view_camera_infos(img_draw, ''.join([f'{IDX}']), int(newbox[0]), int(newbox[1]) - 10)

        KEY = cv2.waitKey(1)

        # quit on ESC button
        if KEY & 0xFF == 27:  # Esc pressed
            cap.release()
            cv2.destroyAllWindows()
            return
        elif KEY == ord('e'):
            break
        elif KEY == ord('c'):
            print('capture start')
            start_capture = 1

        if start_capture == 1:
            CNT += 1
            dump_data(CNT, img_gray, bboxes, boxes, (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT))

        cv2.imshow('MultiTracker', img_draw)

    cap.release()
    cv2.destroyAllWindows()

    bboxes.clear()



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


if __name__ == '__main__':
    print('make dataset auto')
    cam_dev_array = []
    cam_dev_list = terminal_cmd('v4l2-ctl', '--list-devices')
    for i in range(len(cam_dev_list)):
        cam_info = cam_dev_list[i].split('\n\t')
        cam_dev_array.append(cam_info[1])
    print('cam_info', cam_dev_array)
    make_dataset_tracker(cam_dev_array[0])



