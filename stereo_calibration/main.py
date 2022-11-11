import matplotlib.pyplot as plt
import numpy as np
import qdarkstyle

from definition import *
from stereo_camera import *

import os
import sys
import usb.core
import usb.backend.libusb1
import numpy as np
import open3d as o3d

# Test Code for stereo camera calibration system

if __name__ == '__main__':
    cam_dev_list = terminal_cmd('v4l2-ctl', '--list-devices')
    leds_dic['cam_info'] = init_model_json(cam_dev_list)
    leds_dic['pts'] = init_coord_json(ORIGIN)

    # ToDo make stereo camera and calibration
    # camera_setting()
    # camera_rt_test()
    # stereo_calibrate()
    stereo_camera_start()

