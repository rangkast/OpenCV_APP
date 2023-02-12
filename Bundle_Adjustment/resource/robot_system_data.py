import copy

import numpy as np

from .definition import *


@dataclass
class BLOB_2D:
    idx: int = -1
    x: float = -1.0
    y: float = -1.0


@dataclass
class BLOB_3D:
    idx: int = -1
    x: float = -1.0
    y: float = -1.0
    z: float = -1.0


@dataclass
class BLOB_DATA:
    curr_blobs: np.ndarray
    med_blobs: np.ndarray
    acc_blobs: np.ndarray
    distort_2d: np.ndarray
    undistort_2d: np.ndarray

    def get_curr_blobs(self):
        return self.curr_blobs

    def get_med_blobs(self):
        return self.med_blobs

    def get_acc_blobs(self):
        return self.acc_blobs

    def get_distort_2d(self):
        return self.distort_2d

    def get_undistort_2d(self):
        return self.undistort_2d

    def set_curr_blobs(self, data):
        self.curr_blobs = copy.deepcopy(data)

    def set_med_blobs(self, data):
        self.med_blobs = copy.deepcopy(data)

    def set_acc_blobs(self, data):
        self.acc_blobs = np.append(self.acc_blobs, data)

    # list slicing
    def set_distort_2d(self, data):
        self.distort_2d = data[:]

    def set_undistort_2d(self, data):
        self.undistort_2d = data[:]


@dataclass
class CAMERA_CALIBRATION_DATA:
    cameraK: np.ndarray
    distCoeff: np.ndarray


@dataclass
class R_T:
    rvecs: np.ndarray
    tvecs: np.ndarray


@dataclass
class R_T_DATA:
    curr_r_t: np.ndarray
    med_r_t: np.ndarray
    acc_r_t: np.ndarray

    def get_curr_r_t(self):
        return self.curr_r_t

    def get_acc_r_t(self):
        return self.acc_r_t

    def get_med_r_t(self):
        return self.med_r_t

    def set_acc_r_t(self, data):
        self.acc_r_t = np.append(self.acc_r_t, data)

    def set_curr_r_t(self, data):
        self.curr_r_t = copy.deepcopy(data)

    def set_med_r_t(self, data):
        self.med_r_t = copy.deepcopy(data)


@dataclass
class DISPLAY_DATA:
    width: int
    height: int
    rotate: int


@dataclass
class BT_DATA:
    mac: str
    name: str
    list: Dict[str, int]


@dataclass
class SYSTEM_SETTING_DATA:
    mode: np.ndarray = np.array([-1, -1])
    system_name: str = NOT_SET
    camera_json: np.ndarray = NOT_SET
    function_setting: Dict[str, int] = NOT_SET
    solve_pnp: POSE_ESTIMATION_METHOD = NOT_SET
    solve_pnp_distortion: int = -1
    lr_position: int = -1
    bt_data: BT_DATA = BT_DATA(NOT_SET, NOT_SET, NOT_SET)
    led_cnt: int = -1

    def get_mode(self):
        return self.mode[0]

    def get_camera_mode(self):
        return self.mode[1]

    def get_system_name(self):
        return self.system_name

    def get_camera_json(self):
        return self.camera_json

    def get_functions(self):
        return self.function_setting

    def check_functions(self, name):
        for FUNCTIONS, STATUS in self.function_setting.items():
            if STATUS == ENABLE and name == FUNCTIONS.__name__:
                return SUCCESS
        return ERROR

    def get_bt_data(self):
        return self.bt_data

    def get_led_cnt(self):
        return self.led_cnt

    def get_solve_pnp_setting(self):
        return self.solve_pnp, self.solve_pnp_distortion

    def get_lr_position(self):
        return self.lr_position

    def set_bt_data(self, data):
        self.bt_data = copy.deepcopy(data)

    def set_led_cnt(self, data):
        self.led_cnt = data

    def print_all(self):
        print('system setting mode:', self.system_setting_mode)
        print('system_name:', self.system_name)
        print('cam_json:', self.camera_json)
        print('functions:', self.function_setting)


@dataclass
class CAMERA_INFO_DATA:
    dev: str = NOT_SET
    name: str = NOT_SET
    idx: int = -1
    port: str = NOT_SET
    display: DISPLAY_DATA = DISPLAY_DATA(CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, DEGREE_0)
    camera_cal: CAMERA_CALIBRATION_DATA = CAMERA_CALIBRATION_DATA(default_cameraK, default_distCoeff)

    def get_dev(self):
        return self.dev

    def get_name(self):
        return self.name

    def get_port(self):
        return self.port

    def get_id(self):
        return self.idx

    def get_display_info(self):
        return self.display.width, self.display.height, self.display.rotate

    def get_camera_calibration(self):
        return self.camera_cal.cameraK, self.camera_cal.distCoeff

    def set_display_info(self, w, h, r):
        self.display = DISPLAY_DATA(w, h, r)

    def print_all(self):
        print('camera_info_data')
        print('id:', self.idx, ' ', self.dev, ' ', self.name, ' ', self.port)
        print('display:', self.display.width, 'x', self.display.height, ' ', self.display.rotate)
        print('cameraK:', self.camera_cal.cameraK)
        print('distCoeff:', self.camera_cal.distCoeff)


@dataclass
class MEASUREMENT_DATA:
    blob: BLOB_DATA = BLOB_DATA(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]))
    r_t: R_T_DATA = R_T_DATA(np.array([]), np.array([]), np.array([]))

    def get_blob(self):
        return copy.deepcopy(self.blob)

    def get_r_t(self):
        return copy.deepcopy(self.r_t)

    def set_blob(self, data):
        self.blob = copy.deepcopy(data)

    def set_r_t(self, data):
        self.r_t = copy.deepcopy(data)


@dataclass
class REMAKE_3D:
    cam_l: int
    cam_r: int
    blob: BLOB_3D


@dataclass
class GROUP_DATA:
    pair_xy: np.ndarray = np.array([])
    remake_3d: REMAKE_3D = np.array([])

    def get_pair_xy(self):
        return self.pair_xy

    def get_remake_3d(self):
        return self.remake_3d

    def set_pair_xy(self, data):
        self.pair_xy = np.append(self.pair_xy, data)

    def set_remake_3d(self, data):
        self.remake_3d = copy.deepcopy(data)


@dataclass
class LED_DATA:
    idx: int = -1
    pos: np.ndarray = np.array([])
    dir: np.ndarray = np.array([])
    res: np.ndarray = np.array([])
    remake_3d: REMAKE_3D = np.array([])

    def get_idx(self):
        return self.idx

    def get_pos(self):
        return self.pos

    def get_dir(self):
        return self.dir

    def get_res(self):
        return self.res

    def get_remake_3d(self):
        return self.remake_3d

    def set_remake_3d(self, data):
        self.remake_3d = np.append(self.remake_3d, data)

    def print_all(self):
        print('idx:', self.idx, ' ', self.pos, ' ', self.dir)
        print('pair xy:', self.pair_xy)
        print('remake 3d:', self.remake_3d)