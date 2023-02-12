import time

from robot_system.connection.socket.socket_def import *

def move_x(val):
    current_t[TVEC_INDEX.x] += val
    return "ok"

def move_y(val):
    current_t[TVEC_INDEX.y] += val
    return "ok"

def move_z(val):
    current_t[TVEC_INDEX.z] += val
    return "ok"

def rotate_Rall(val):
    current_r[RVEC_INDEX.Rall] += val
    return "ok"

def rotate_Pitch(val):
    current_r[RVEC_INDEX.Pitch] += val
    return "nok"

def rotate_Yaw(val):
    current_r[RVEC_INDEX.Yaw] += val
    return "ok"

def reset_location():
    current_t[TVEC_INDEX.x] = 30
    current_t[TVEC_INDEX.y] = 30
    current_t[TVEC_INDEX.z] = 30
    current_r[RVEC_INDEX.Rall] = 0
    current_r[RVEC_INDEX.Pitch] = 0
    current_r[RVEC_INDEX.Yaw] = 0
    return "ROSuccess"

def robot_operation():
    # print(rt_list)
    # print(rt_val_list)
    for idx, rt_cmd in enumerate(rt_list):
        if rt_cmd == 'x':
            op_check = move_x(rt_val_list[idx])
        elif rt_cmd == 'y':
            op_check = move_y(rt_val_list[idx])
        elif rt_cmd == 'z':
            op_check = move_z(rt_val_list[idx])
        elif rt_cmd == 'R':
            op_check = rotate_Rall(rt_val_list[idx])
        elif rt_cmd == 'P':
            op_check = rotate_Pitch(rt_val_list[idx])
        elif rt_cmd == 'Y':
            op_check = rotate_Yaw(rt_val_list[idx])
        if op_check == 'nok':
            return "ROFail"
    return "ROSuccess"
