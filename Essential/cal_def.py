from dataclasses import dataclass

import copy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np


@dataclass
class vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __array__(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def clamp(self, mmin, mmax):
        self.x = max(mmin, self.x)
        self.x = min(mmax, self.x)
        self.y = max(mmin, self.y)
        self.y = min(mmax, self.y)
        self.z = max(mmin, self.z)
        self.z = min(mmax, self.z)

    def mult_me(self, d):
        self.x *= d
        self.y *= d
        self.z *= d

    def normalize_me(self):
        if self.x == 0 and self.y == 0 and self.z == 0:
            return
        len = self.get_length()
        # print('1: ', self.x, ' ', self.y, ' ', self.z, ' ', len)
        self.x /= len
        self.y /= len
        self.z /= len
        # print('2: ', self.x, ' ', self.y, ' ', self.z, ' ', len)
        return vector3(self.x, self.y, self.z)

    def get_length(self):
        return np.sqrt(np.sum(np.power(np.array(self), 2)))

    def copy(self):
        return vector3(self.x, self.y, self.z)

    def round(self, d):
        self.x = round(self.x, d)
        self.y = round(self.y, d)
        self.z = round(self.z, d)

    def round_(self, d):
        return vector3(round(self.x, d), round(self.y, d), round(self.z, d))

    def get_rotated(self, tq):
        q = quat(self.x * tq.w + self.z * tq.y - self.y * tq.z,
                 self.y * tq.w + self.x * tq.z - self.z * tq.x,
                 self.z * tq.w + self.y * tq.x - self.x * tq.y,
                 self.x * tq.x + self.y * tq.y + self.z * tq.z)
        return vector3(tq.w * q.x + tq.x * q.w + tq.y * q.z - tq.z * q.y,
                       tq.w * q.y + tq.y * q.w + tq.z * q.x - tq.x * q.z,
                       tq.w * q.z + tq.z * q.w + tq.x * q.y - tq.y * q.x)

    def add_vector3(self, t):
        return vector3(round(self.x + t.x, 8), round(self.y + t.y, 8), round(self.z + t.z, 8))

    def sub_vector3(self, t):
        return vector3(round(self.x - t.x, 8), round(self.y - t.y, 8), round(self.z - t.z, 8))

    def get_dot(self, t):
        return self.x * t.x + self.y * t.y + self.z * t.z


@dataclass
class quat:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def __array__(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.w])

    def mult_me_quat(self, q):
        tmp = self.copy()
        self.x = tmp.w * q.x + tmp.x * q.w + tmp.y * q.z - tmp.z * q.y
        self.y = tmp.w * q.y - tmp.x * q.z + tmp.y * q.w + tmp.z * q.x
        self.z = tmp.w * q.z + tmp.x * q.y - tmp.y * q.x + tmp.z * q.w
        self.w = tmp.w * q.w - tmp.x * q.x - tmp.y * q.y - tmp.z * q.z

    def mult_quat(self, q):
        tmp = self.copy()
        return quat(tmp.w * q.x + tmp.x * q.w + tmp.y * q.z - tmp.z * q.y,
                    tmp.w * q.y - tmp.x * q.z + tmp.y * q.w + tmp.z * q.x,
                    tmp.w * q.z + tmp.x * q.y - tmp.y * q.x + tmp.z * q.w,
                    tmp.w * q.w - tmp.x * q.x - tmp.y * q.y - tmp.z * q.z)

    def copy(self):
        return quat(self.x, self.y, self.z, self.w)

    '''
    def inverse(self):
        dot = quat_get_dot(self, self)
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        self.mult_me(1 / dot)
    '''

    def round(self, d):
        self.x = round(self.x, d)
        self.y = round(self.y, d)
        self.z = round(self.z, d)
        self.w = round(self.w, d)

    def round_(self, d):
        return quat(round(self.x, d), round(self.y, d), round(self.z, d), round(self.w, d))

    def inverse(self):
        dot = quat_get_dot(self, self)
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        self.x = self.x / dot
        self.y = self.y / dot
        self.z = self.z / dot
        self.w = self.w / dot


def quat_get_dot(self, t):
    return self.x * t.x + self.y * t.y + self.z * t.z + self.w * t.w


def get_dot_point(pt, t):
    return pt.get_dot(t)


def nomalize_point(pt):
    return pt.normalize_me()


def rotate_point(pt, pose):
    return pt.get_rotated(pose['orient'])


def transfer_point(pt, pose):
    r_pt = pt.get_rotated(pose['orient'])
    return r_pt.add_vector3(pose['position'])


def transfer_point_(pt, pose):
    r_pt = pt.get_rotated(pose['reorient'])
    return r_pt.add_vector3(pose['position'])


def transfer_point_inverse(pt, pose):
    t = copy.deepcopy(pose)
    t['orient'].inverse()
    r_pt = pt.sub_vector3(t['position'])
    return r_pt.get_rotated(t['orient'])


def get_quat_from_euler(order, value):
    rt = R.from_euler(order, value, degrees=True)
    return quat(rt.as_quat()[0], rt.as_quat()[1], rt.as_quat()[2], rt.as_quat()[3])


def pose_apply(a, b):
    return {'position': transfer_point(a['position'], b), 'orient': b['orient'].mult_quat(a['orient'])}


def pose_apply_inverse(a, b):
    t = copy.deepcopy(b)
    t['orient'].inverse()
    tmp = a['position'].sub_vector3(t['position'])
    return {'position': tmp.get_rotated(t['orient']), 'orient': t['orient'].mult_quat(a['orient'])}


def get_euler_from_quat(order, q):
    rt = R.from_quat(np.array(q))
    return rt.as_euler(order, degrees=True)


def unit_vector(vector):
    """Returnstheunitvectorofthevector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returnstheangleinradiansbetweenvectors'v1'and'v2'::
    >>>angle_between((1,0,0),(0,1,0))
    1.5707963267948966
    >>>angle_between((1,0,0),(1,0,0))
    0.0
    >>>angle_between((1,0,0),(-1,0,0))
    3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_2(v1, v2):
    unit_vector1 = v1 / np.linalg.norm(v1)
    unit_vector2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    return np.arccos(dot_product)


def zoom_factory(ax, base_scale=2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange * scale_factor,
                     xdata + cur_xrange * scale_factor])
        ax.set_ylim([ydata - cur_yrange * scale_factor,
                     ydata + cur_yrange * scale_factor])
        plt.draw()  # force re-draw

    fig = ax.get_figure()  # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event', zoom_fun)
    # return the function
    return zoom_fun


def add_random_normal():
    offset = np.random.normal(avg, sd, 3)
    for i, v in enumerate(offset):
        offset[i] = np.round_(np.clip(v, -sd, sd), 6)
    return offset


def world2cam(coord, cam_pose):
    ret_coord = transfer_point(coord, cam_pose)
    print('world2cam:', ret_coord)
    return ret_coord


def camtoworld(coord, cam_pose):
    pt = copy.deepcopy(cam_pose)
    pt['position'] = vector3(0, 0, 0)
    new_pt = transfer_point(coord, pt)
    return new_pt


def camtoworld_(coord, cam_pose):
    pt = copy.deepcopy(cam_pose)
    pt['position'] = vector3(0, 0, 0)
    new_pt = transfer_point_(coord, pt)
    return new_pt


# add common function here
def draw_dots(dimension, pts, ax, c):
    if dimension == 3:
        x = [x['pos'][0] for x in pts]
        y = [x['pos'][1] for x in pts]
        z = [x['pos'][2] for x in pts]
        u = [x['dir'][0] for x in pts]
        v = [x['dir'][1] for x in pts]
        w = [x['dir'][2] for x in pts]
        idx = [x['idx'] for x in pts]
        ax.scatter(x, y, z, marker='o', s=5, color=c, alpha=0.5)
        ax.quiver(x, y, z, u, v, w, length=0.05, linewidths=0.1, color='red', normalize=True)
        for idx, x, y, z in zip(idx, x, y, z):
            label = '%s' % idx
            ax.text(x, y, z, label, size=5)

    elif dimension == 2:
        x = [x['pos'][0] for x in pts]
        y = [x['pos'][1] for x in pts]
        idx = [x['idx'] for x in pts]
        ax.scatter(x, y, marker='o', s=5, color=c, alpha=0.5)
        # 좌표 numbering
        if c == 'red':
            for idx, x, y in zip(idx, x, y):
                label = '%s' % idx
                ax.text(x + 0.001, y + 0.001, label, size=7)


########################################### DEFINE #####################################################
ENABLE = 1
DISABLE = 0
# offset value
offset = {'position': vector3(0.003, 0.003, 0.003), 'orient': get_quat_from_euler('xyz', [1, 1, 1])}
# offset = {'position': vector3(0.00, 0.00, 0.00), 'orient': get_quat_from_euler('xyz', [0, 0, 0])}

# 추가 된 카메라 정보
camera_array = []
# 카메라 별 변환된 좌표 array
trans_leds_array = []
trans_noise_leds_array = []

angle_spec = 65
leds_dic = {'name': 'led_dictionary'}

# data set
pts = []
pts_noise = []

# 여기에 역산으로 구한 데이터를 넣어보자.
pts_refactor = []
pts_noise_refactor = []
remake_offset_arr = []

#
# cameraK = np.eye(3)
# distCoeff = np.zeros((5, 1))


cameraK = np.array([[712.623, 0.0, 653.448],
                    [0.0, 712.623, 475.572],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
distCoeff = np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)

# features
draw_camera_point = DISABLE
draw_facing_dot = DISABLE

insert_coord_noise_led = ENABLE
insert_dir_noise_led = DISABLE
add_pose_offset = ENABLE
avg = 0
sd = 0.001
# sd = 0.0

debug = ENABLE

do_sansac = ENABLE
# Camera Section
draw_camera_pos = ENABLE
camera_section_1 = ENABLE
camera_section_2 = ENABLE
camera_section_3 = ENABLE

camera_half_section = ENABLE
# 순서 맞아야 함!!!
# section 1
# if camera_section_1 == ENABLE:
#     camera_array.append({'idx': 0, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [0, -90, 0]), 'reorient': get_quat_from_euler('xyz', [0, -90, 0])})
#     camera_array.append({'idx': 1, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [0, 90, 0]), 'reorient': get_quat_from_euler('xyz', [0, 90, 0])})
#     camera_array.append({'idx': 2, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [90, 0, 0]), 'reorient': get_quat_from_euler('xyz', [90, 0, 0])})
#     camera_array.append({'idx': 3, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [-90, 0, 0]), 'reorient': get_quat_from_euler('xyz', [-90, 0, 0])})
#     if camera_half_section == ENABLE:
#         camera_array.append({'idx': 4, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [90, 45, 0]), 'reorient': get_quat_from_euler('xyz', [45, 90, 0])})
#         camera_array.append({'idx': 5, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [90, -45, 0]), 'reorient': get_quat_from_euler('xyz', [45, -90, 0])})
#         camera_array.append({'idx': 6, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [-90, 45, 0]), 'reorient': get_quat_from_euler('xyz', [-45, 90, 0])})
#         camera_array.append({'idx': 7, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [-90, -45, 0]), 'reorient': get_quat_from_euler('xyz', [-45, -90, 0])})
#
# # section 2
# if camera_section_2 == ENABLE:
#     camera_array.append({'idx': 8, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [0, -135, 0]), 'reorient': get_quat_from_euler('xyz', [0, -135, 0])})
#     camera_array.append({'idx': 9, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [0, 135, 0]), 'reorient': get_quat_from_euler('xyz', [0, 135, 0])})
#     camera_array.append({'idx': 10, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [135, 0, 0]), 'reorient': get_quat_from_euler('xyz', [135, 0, 0])})
#     camera_array.append({'idx': 11, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [-135, 0, 0]), 'reorient': get_quat_from_euler('xyz', [-135, 0, 0])})
#     if camera_half_section == ENABLE:
#         camera_array.append({'idx': 12, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [135, 45, 0]), 'reorient': get_quat_from_euler('xyz', [45, 135, 0])})
#         camera_array.append({'idx': 13, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [135, -45, 0]), 'reorient': get_quat_from_euler('xyz', [45, -135, 0])})
#         camera_array.append({'idx': 14, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [-135, 45, 0]), 'reorient': get_quat_from_euler('xyz', [-45, 135, 0])})
#         camera_array.append({'idx': 15, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [-135, -45, 0]), 'reorient': get_quat_from_euler('xyz', [-45, -135, 0])})
#
# # section 3
# if camera_section_3 == ENABLE:
#     camera_array.append({'idx': 16, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [0, -45, 0]), 'reorient': get_quat_from_euler('xyz', [0, -45, 0])})
#     camera_array.append({'idx': 17, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [0, 45, 0]), 'reorient': get_quat_from_euler('xyz', [0, 45, 0])})
#     camera_array.append({'idx': 18, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [45, 0, 0]), 'reorient': get_quat_from_euler('xyz', [45, 0, 0])})
#     camera_array.append({'idx': 19, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [-45, 0, 0]), 'reorient': get_quat_from_euler('xyz', [-45, 0, 0])})
#     if camera_half_section == ENABLE:
#         camera_array.append({'idx': 20, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [45, 45, 0]), 'reorient': get_quat_from_euler('xyz', [45, 45, 0])})
#         camera_array.append({'idx': 21, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [45, -45, 0]), 'reorient': get_quat_from_euler('xyz', [45, -45, 0])})
#         camera_array.append({'idx': 22, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [-45, 45, 0]), 'reorient': get_quat_from_euler('xyz', [-45, 45, 0])})
#         camera_array.append({'idx': 23, 'position': vector3(0, 0, 0.3), 'orient': get_quat_from_euler('xyz', [-45, -45, 0]), 'reorient': get_quat_from_euler('xyz', [-45, -45, 0])})
# cam_len = len(camera_array)

########################################### DEFINE #####################################################
