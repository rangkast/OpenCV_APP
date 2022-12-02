# init camera data array
import sys
from definition import *
from phone_camera import *

if __name__ == '__main__':
    for p in sys.path:
        print(p)
    cam_dev_list = terminal_cmd('v4l2-ctl', '--list-devices')
    leds_dic['cam_info'] = init_data_array(cam_dev_list)
    leds_dic['pts'] = init_coord_json(ORIGIN)
    leds_dic['target_pts'] = init_coord_json(TARGET)

    # camera_setting_test()
    camera_rt_test()




