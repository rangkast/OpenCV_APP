import matplotlib.pyplot as plt
import numpy as np
import qdarkstyle

from definition import *
from camera import *
import os
import sys
import usb.core
import usb.backend.libusb1
from uvc_openCV import *

if __name__ == '__main__':
    print('start advanced inlier loop calibration')

    cam_dev_list = terminal_cmd('v4l2-ctl', '--list-devices')

    leds_dic['test_status'] = init_test_status()
    leds_dic['cam_info'] = init_model_json(cam_dev_list)

    leds_dic['pts'] = init_coord_json(ORIGIN)
    leds_dic['target_pts'] = init_coord_json(TARGET)
    init_rt_custom_led_filter()
    # camera_single_test()
    multi_tracker()
