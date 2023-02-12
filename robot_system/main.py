from robot_system.connection.bluetooth import *
from robot_system.system import *

TAG = '[MAIN]'


# SET MAIN FUNCTIONS STATUS
MAIN_FUNCTIONS_SETTING = {
    # Default
    robot_init_data_arrays: ENABLE,
    # Setting
    robot_camera_setting: ENABLE,
    robot_camera_default: DISABLE,
    robot_module_lsm: DISABLE,
    robot_print_result: DISABLE,
    robot_bt_send: DISABLE,

    # TEST CODE
    robot_animate_tracker: ENABLE,
    robot_dump_data: ENABLE,
}

# SYSTEM SETTING DATA
ROBOT_SYSTEM_DATA[SYSTEM_SETTING] = SYSTEM_SETTING_DATA(
    # Set SYSTEM_SETTING_MODE or CALIBRATION_MODE
    # Set Camera MODE
    [MODE.CALIBRATION_MODE, CAMERA_MODE.DUAL_CAMERA],
    # Camera Sensor Setting
    SENSOR_NAME_RIFT, cam_json_rift,
    # Functions Setting
    MAIN_FUNCTIONS_SETTING,
    # Set solvePnP Setting
    POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC, UNDISTORTION,
    # Set controller LR position
    LR_POSITION.LEFT,
)

if __name__ == '__main__':
    print(TAG, 'START Robot Calibration System')
    # Main Functions Call
    for FUNCTIONS, STATUS in ROBOT_SYSTEM_DATA[SYSTEM_SETTING].get_functions().items():
        ret = FUNCTIONS() if STATUS == ENABLE else print(TAG, FUNCTIONS.__name__, 'SKIP')
        if ret == ERROR:
            break

# end main
