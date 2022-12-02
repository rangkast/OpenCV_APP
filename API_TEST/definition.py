import json
import random
import signal
from collections import OrderedDict
from dataclasses import dataclass
import copy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
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


def move_point(pt, pose):
    return pt.add_vector3(pose)


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


def rw_json_data(rw_mode, path, data):
    try:
        if rw_mode == READ:
            with open(path, 'r', encoding="utf-8") as rdata:
                json_data = json.load(rdata)
            return json_data
        elif rw_mode == WRITE:
            with open(path, 'w', encoding="utf-8") as wdata:
                json.dump(data, wdata, ensure_ascii=False, indent="\t")
        else:
            print('not support mode')
    except:
        print('exception')
        return ERROR


def rw_file_storage(rw_cmd, name, left_map, right_map):
    if rw_cmd == WRITE:
        print("WRITE parameters ......")
        cv_file = cv2.FileStorage(name, cv2.FILE_STORAGE_WRITE)
        cv_file.write("Left_Stereo_Map_x", left_map[0])
        cv_file.write("Left_Stereo_Map_y", left_map[1])
        cv_file.write("Right_Stereo_Map_x", right_map[0])
        cv_file.write("Right_Stereo_Map_y", right_map[1])
        cv_file.release()
    else:
        print("READ parameters ......")
        try:
            # FILE_STORAGE_READ
            cv_file = cv2.FileStorage(name, cv2.FILE_STORAGE_READ)
            # note we also have to specify the type to retrieve other wise we only get a
            # FileNode object back instead of a matrix
            left_map = (cv_file.getNode("Left_Stereo_Map_x").mat(), cv_file.getNode("Left_Stereo_Map_y").mat())
            right_map = (cv_file.getNode("Right_Stereo_Map_x").mat(), cv_file.getNode("Right_Stereo_Map_y").mat())

            cv_file.release()

            return DONE, left_map, right_map
        except:
            traceback.print_exc()
            return ERROR, NOT_SET, NOT_SET


def Rotate(src, degrees):
    if degrees == 90:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 1)

    elif degrees == 180:
        dst = cv2.flip(src, -1)

    elif degrees == 270:
        dst = cv2.transpose(src)
        dst = cv2.flip(dst, 0)
    else:
        dst = src
    return dst


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
        if DEBUG == ENABLE:
            print(devices)
    temp = result.split('\n\n')
    print("==================================================")
    ret_val = []
    for i in range(len(temp)):
        if SENSOR_NAME in temp[i]:
            ret_val.append(temp[i])
            print("add camera dev", temp[i])
        else:
            print("skipping camera", temp[i])
    print("==================================================")
    return ret_val


def init_data_array(cam_dev_list):
    print(cam_dev_list)
    camera_info_array = []
    for i in range(len(cam_dev_list)):
        cam_info = cam_dev_list[i].split('\n\t')

        temp_struct = copy.deepcopy(cam_info_struct)

        if SENSOR_NAME == 'Droidcam':
            for cam_id, dev_name in enumerate(cam_info):
                if 'dev' in dev_name:
                    # print(dev_name)
                    temp_struct['idx'] = cam_id
                    temp_struct['port'] = dev_name
        elif SENSOR_NAME == 'Rift':
            temp_struct['idx'] = i
            temp_struct['port'] = cam_info[1]

            print('cam_id[', i, '] :', cam_json[i])
            jsonObject = json.load(open(''.join(['jsons/', f'{cam_json[i]}'])))

            cam_info = cam_dev_list[i].split('\n\t')

            '''                        
              k = [ k₁ k₂, k₃, k4 ] for CV1 fisheye distortion                    

                  ⎡ fx 0  cx ⎤
              A = ⎢ 0  fy cy ⎥
                  ⎣ 0  0  1  ⎦          
            '''
            f = jsonObject.get('camera_f')
            c = jsonObject.get('camera_c')
            k = jsonObject.get('camera_k')

            A = np.array([[f[0], 0.0, c[0]],
                          [0.0, f[1], c[1]],
                          [0.0, 0.0, 1.0]], dtype=np.float64)
            K = np.array([[k[0]], [k[1]], [k[2]], [k[3]]], dtype=np.float64)

            if DEBUG == ENABLE:
                print('cameraK: ', A)
                print('dist_coeff: ', K)

            temp_struct['cam_cal']['cameraK'] = A
            temp_struct['cam_cal']['dist_coeff'] = K

        camera_info_array.append(temp_struct)
    print('camera info', camera_info_array)

    return camera_info_array


def init_coord_json(file):
    print('start ', init_coord_json.__name__)
    try:
        json_file = open(''.join(['jsons/specs/', f'{file}']))
        jsonObject = json.load(json_file)
        model_points = jsonObject.get('TrackedObject').get('ModelPoints')
        pts = [0 for i in range(len(model_points))]
        for data in model_points:
            idx = data.split('Point')[1]
            x = model_points.get(data)[0]
            y = model_points.get(data)[1]
            z = model_points.get(data)[2]
            u = model_points.get(data)[3]
            v = model_points.get(data)[4]
            w = model_points.get(data)[5]
            r1 = model_points.get(data)[6]
            r2 = model_points.get(data)[7]
            r3 = model_points.get(data)[8]
            pts[int(idx)] = {'idx': idx,
                             'pos': [x, y, z],
                             'dir': [u, v, w],
                             'res': [r1, r2, r3],
                             'pair_xy': [],
                             'remake_3d': []}

            print(''.join(['{ .pos = {{', f'{x}', ',', f'{y}', ',', f'{z}',
                           ' }}, .dir={{', f'{u}', ',', f'{v}', ',', f'{w}', ' }}, .pattern=', f'{idx}', '},']))
    except:
        print('exception')
        traceback.print_exc()
    finally:
        print('done')
    return pts


def view_camera_infos(frame, text, x, y):
    cv2.putText(frame, text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), lineType=cv2.LINE_AA)


def find_center(frame, led_num, X, Y, W, H, blobs):
    x_sum = 0
    t_sum = 0
    y_sum = 0
    m_count = 0
    g_c_x = 0
    g_c_y = 0

    ret_blobs = copy.deepcopy(blobs)

    for y in range(Y, Y + H):
        for x in range(X, X + W):
            if frame[y][x] >= CV_MID_THRESHOLD:
                x_sum += x * frame[y][x]
                t_sum += frame[y][x]
                m_count += 1

    for x in range(X, X + W):
        for y in range(Y, Y + H):
            if frame[y][x] >= CV_MID_THRESHOLD:
                y_sum += y * frame[y][x]

    if t_sum != 0:
        g_c_x = x_sum / t_sum
        g_c_y = y_sum / t_sum

    # print('led ', led_num, ' x ', g_c_x, ' y ', g_c_y)

    if g_c_x == 0 or g_c_y == 0:
        return ERROR

    if len(ret_blobs) > 0:
        detect = 0
        for i, datas in enumerate(ret_blobs):
            led = datas['idx']
            if led == led_num:
                ret_blobs[i] = {'idx': led_num, 'cx': g_c_x, 'cy': g_c_y}
                detect = 1
                break
        if detect == 0:
            ret_blobs.append({'idx': led_num, 'cx': g_c_x, 'cy': g_c_y})
    else:
        ret_blobs.append({'idx': led_num, 'cx': g_c_x, 'cy': g_c_y})

    return DONE, ret_blobs


def simple_solvePNP(cam_id, frame, blob_array):
    model_points = []
    image_points = []
    led_ids = []
    interationsCount = 100
    confidence = 0.99

    for blobs in blob_array:
        led_num = int(blobs['idx'])
        # if DEBUG == ENABLE:
        #     print('idx:', led_num, ' added 3d', leds_dic['pts'][led_num]['pos'], ' remake: ',
        #           leds_dic['target_pts'][led_num]['remake_3d'],
        #           ' 2d', [blobs['cx'], blobs['cy']])

        model_points.append(leds_dic['pts'][led_num]['pos'])
        led_ids.append(led_num)
        image_points.append([blobs['cx'], blobs['cy']])

    model_points_len = len(model_points)
    image_points_len = len(image_points)

    # check assertion
    if model_points_len != image_points_len:
        print("assertion len is not equal")
        return ERROR

    if model_points_len < 4 or image_points_len < 4:
        print("assertion < 4: ")
        return ERROR

    camera_k = leds_dic['cam_info'][cam_id]['cam_cal']['cameraK']
    dist_coeff = leds_dic['cam_info'][cam_id]['cam_cal']['dist_coeff']

    list_2d_distorted = np.zeros((image_points_len, 1, 2), dtype=np.float64)
    for i in range(image_points_len):
        list_2d_distorted[i] = image_points[i]

    points3D = np.array(model_points)

    list_2d_undistorted = cv2.fisheye.undistortPoints(list_2d_distorted, camera_k, dist_coeff)
    leds_dic['cam_info'][cam_id]['distorted_2d'] = copy.deepcopy(list_2d_distorted)
    leds_dic['cam_info'][cam_id]['undistorted_2d'] = copy.deepcopy(list_2d_undistorted)

    if DO_UNDISTORT == ENABLE:
        temp_points2D = []
        for u_data in list_2d_undistorted:
            temp_points2D.append([u_data[0][0], u_data[0][1]])
        points2D = np.array(temp_points2D)
        temp_camera_k = cameraK
        temp_dist_coeff = distCoeff
    else:
        points2D = np.array(image_points)
        temp_camera_k = leds_dic['cam_info'][cam_id]['cam_cal']['cameraK']
        temp_dist_coeff = leds_dic['cam_info'][cam_id]['cam_cal']['dist_coeff']

    if DO_SOLVEPNP_RANSAC == ENABLE:
        success, rvecs, tvecs, inliers = cv2.solvePnPRansac(points3D, points2D,
                                                            temp_camera_k,
                                                            temp_dist_coeff,
                                                            useExtrinsicGuess=True,
                                                            iterationsCount=interationsCount,
                                                            confidence=confidence,
                                                            reprojectionError=1.0,
                                                            flags=cv2.SOLVEPNP_ITERATIVE)

    else:
        success, rvecs, tvecs = cv2.solvePnP(points3D, points2D,
                                             temp_camera_k,
                                             temp_dist_coeff,
                                             flags=cv2.SOLVEPNP_AP3P)
        # length = len(points2D)
        # for i in range(length):
        #     inliers = np.array(
        #         [i for i in range(length)]).reshape(
        #         length, 1)

    ret, RER, TRER, candidate_array, except_inlier = cal_RER_px(led_ids, frame,
                                                                points3D, points2D,
                                                                inliers,
                                                                rvecs,
                                                                tvecs,
                                                                temp_camera_k,
                                                                temp_dist_coeff, DONE)

    # if ret == SUCCESS:
    #     #####
    #     leds_dic['cam_info'][cam_id]['D_R_T_A'].append({'rvecs': rvecs, 'tvecs': tvecs})
    #     #####
    #     leds_dic['cam_info'][cam_id]['RER']['C_R_T'] = {'rvecs': rvecs, 'tvecs': tvecs}
    #     return SUCCESS, candidate_array
    # else:
    #     return ERROR, candidate_array

    leds_dic['cam_info'][cam_id]['D_R_T_A'].append({'rvecs': rvecs, 'tvecs': tvecs})
    #####
    leds_dic['cam_info'][cam_id]['RER']['C_R_T'] = {'rvecs': rvecs, 'tvecs': tvecs}
    return SUCCESS, blob_array


def cal_RER_px(led_ids, frame, points3D, points2D, inliers, rvecs, tvecs, camera_k, dist_coeff, status):
    # Compute re-projection error.
    blob_array = []
    points2D_reproj = cv2.projectPoints(points3D, rvecs,
                                        tvecs, camera_k, dist_coeff)[0].squeeze(1)
    # print('points2D_reproj\n', points2D_reproj, '\npoints2D\n', points2D, '\n inliers: ', inliers)
    assert (points2D_reproj.shape == points2D.shape)
    error = (points2D_reproj - points2D)[inliers]  # Compute error only over inliers.
    # print('error', error)
    rmse = 0.0
    dis = 0.0
    led_except = -1
    except_inlier = -1
    led_dis = []

    for idx, error_data in enumerate(error[:, 0]):
        rmse += np.power(error_data[0], 2) + np.power(error_data[1], 2)
        temp_dis = np.power(error_data[0], 2) + np.power(error_data[1], 2)
        led_dis.append(temp_dis)

        if status == NOT_SET:
            if temp_dis > dis:
                dis = temp_dis
                led_except = led_ids[idx]
                except_inlier = idx

        print('led_num: ', led_ids[idx], ' dis:', '%0.18f' % temp_dis, ' : ',
              points2D_reproj[idx][0], ' ', points2D_reproj[idx][1],
              ' vs ', points2D[idx][0], ' ', points2D[idx][1])

        if temp_dis > 100:
            return ERROR, 0, 0, blob_array, except_inlier

        cv2.circle(frame, (int(points2D_reproj[idx][0]), int(points2D_reproj[idx][1])), 1, color=(255, 255, 0),
                   thickness=-1)
        cv2.circle(frame, (int(points2D[idx][0]), int(points2D[idx][1])), 1, color=(0, 0, 255),
                   thickness=-1)

    trmse = float(rmse - dis)
    # print('trmse : ', trmse, ' rmse : ', rmse)
    if inliers is None:
        return ERROR, -1, -1, blob_array, except_inlier
    RER = round(np.sqrt(rmse) / len(inliers), 18)
    if status == NOT_SET:
        TRER = round(np.sqrt(trmse) / (len(inliers) - 1), 18)
        if led_except == -1:
            return ERROR, RER, TRER, blob_array, except_inlier
    else:
        TRER = round(np.sqrt(trmse) / (len(inliers)), 18)

    for i, idx in enumerate(led_ids):
        if idx != led_except:
            blob_array.append({'idx': led_ids[i],
                               'cx': points2D[i][0], 'cy': points2D[i][1], 'area': 0})

    return SUCCESS, RER, TRER, blob_array, except_inlier


def cal_iqr_func(arr):
    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)

    IQR = Q3 - Q1

    outlier_step = 1.5 * IQR

    lower_bound = Q1 - outlier_step
    upper_bound = Q3 + outlier_step

    mask = np.where((arr > upper_bound) | (arr < lower_bound))

    # print(f"cal_iqr_func!!!!!! lower_bound = {lower_bound} upper_bound ={upper_bound} mask = {mask}")

    return mask


def detect_outliers(idx, blob_array, remove_index_array):
    temp_x = np.array(cal_iqr_func(blob_array[0]))
    temp_y = np.array(cal_iqr_func(blob_array[1]))

    for x in temp_x:
        for xx in x:
            if xx in remove_index_array:
                continue
            else:
                remove_index_array.append(xx)
    for y in temp_y:
        for yy in y:
            if yy in remove_index_array:
                continue
            else:
                remove_index_array.append(yy)

    remove_index_array.sort()

    # print("detect_outliers!!!!!!!!!!!!!!!!!!!!!!!!!!!! remove_index_array", remove_index_array)


def median_blobs(cam_id, blob_array, rt_array):
    blob_cnt = len(blob_array)
    if blob_cnt == 0:
        print('blob_cnt is 0')
        return ERROR
    blob_length = len(blob_array[0])

    med_blobs_array = []
    remove_index_array = []
    med_rt_array = []
    print('cam_id:', cam_id, ' blob_cnt:', blob_cnt)

    for i in range(blob_length):
        med_xy = [[], [], []]
        for ii in range(blob_cnt):
            med_xy[0].append(blob_array[ii][i]['cx'])
            med_xy[1].append(blob_array[ii][i]['cy'])
            # med_xy[2].append(blob_array[ii][i]['area'])
        detect_outliers(blob_array[ii][i]['idx'], med_xy, remove_index_array)

    r_len = len(remove_index_array)
    print(f"median_blobs!!!!! remove_index_array length={r_len}")

    for i in range(blob_length):
        med_xy = [[], [], []]
        for ii in range(blob_cnt):
            med_xy[0].append(blob_array[ii][i]['cx'])
            med_xy[1].append(blob_array[ii][i]['cy'])
            # med_xy[2].append(blob_array[ii][i]['area'])
        # tempx=med_xy[0]
        # print(f"original med_xy[0] = {tempx}")
        count = 0
        for index in remove_index_array:
            med_xy[0].pop(index - count)
            med_xy[1].pop(index - count)
            # med_xy[2].pop(index - count)
            count += 1
        # tempx=med_xy[0]
        # print(f"after pop med_xy[0] = {tempx}")

        mean_med_x = np.mean(med_xy[0])
        mean_med_y = np.mean(med_xy[1])

        med_blobs_array.append({'idx': blob_array[ii][i]['idx'],
                                'cx': mean_med_x,
                                'cy': mean_med_y})

    if rt_array != NOT_SET:
        count = 0
        for index in remove_index_array:
            rt_array.pop(index - count)
            count += 1
        for i in range(len(rt_array)):
            rvt = [[], [], []]
            for x in rt_array[i]['rvecs'][0]:
                rvt[0].append(x)
            for y in rt_array[i]['rvecs'][1]:
                rvt[1].append(y)
            for z in rt_array[i]['rvecs'][2]:
                rvt[2].append(z)
            tvt = [[], [], []]
            for x in rt_array[i]['tvecs'][0]:
                tvt[0].append(x)
            for y in rt_array[i]['tvecs'][1]:
                tvt[1].append(y)
            for z in rt_array[i]['tvecs'][2]:
                tvt[2].append(z)

        mean_rvt = []
        mean_tvt = []
        for i in range(0, 3):
            mean_rvt.append(np.mean(rvt[i]))
            mean_tvt.append(np.mean(tvt[i]))

        med_rt_array = {'rvecs': np.array([[mean_rvt[0]], [mean_rvt[1]], [mean_rvt[2]]], dtype=np.float64),
                        'tvecs': np.array([[mean_tvt[0]], [mean_tvt[1]], [mean_tvt[2]]], dtype=np.float64)}

        len_rt_array = len(rt_array)
        print(f"rt_array_len = {len_rt_array}")
        print(f"med_rt_array = {med_rt_array}")
    print(f"med_blobs_array = {med_blobs_array}")

    blob_array = med_blobs_array

    return blob_array, med_rt_array


# add common function here
def draw_dots(dimension, pts, ax, c):
    if dimension == D3D:
        x = [x['pos'][0] for x in pts]
        y = [x['pos'][1] for x in pts]
        z = [x['pos'][2] for x in pts]
        u = [x['dir'][0] for x in pts]
        v = [x['dir'][1] for x in pts]
        w = [x['dir'][2] for x in pts]
        idx = [x['idx'] for x in pts]
        ax.scatter(x, y, z, marker='o', s=3, color=c, alpha=0.5)
        ax.quiver(x, y, z, u, v, w, length=0.05, linewidths=0.1, color='red', normalize=True)
        for idx, x, y, z in zip(idx, x, y, z):
            label = '%s' % idx
            ax.text(x, y, z, label, size=5)
    elif dimension == D3DT:
        for i in range(len(pts)):
            if pts[i]['remake_3d'] != 'error':
                print(pts[i]['remake_3d'])
                x = [coord['coord'][0][0][0] for coord in pts[i]['remake_3d']]
                y = [coord['coord'][0][0][1] for coord in pts[i]['remake_3d']]
                z = [coord['coord'][0][0][2] for coord in pts[i]['remake_3d']]
                idx = [coord['idx'] for coord in pts[i]['remake_3d']]

                ax.scatter(x, y, z, marker='o', s=3, color=c, alpha=0.5)
                for idx, x, y, z in zip(idx, x, y, z):
                    label = '%s' % idx
                    ax.text(x, y, z, label, size=3)
    elif dimension == D2D:
        x = [x['pos'][0] for x in pts]
        y = [x['pos'][1] for x in pts]
        idx = [x['idx'] for x in pts]
        ax.scatter(x, y, marker='o', s=15, color=c, alpha=0.5)
        # 좌표 numbering
        for idx, x, y in zip(idx, x, y):
            label = '%s' % idx
            ax.text(x + 0.001, y + 0.001, label, size=6)


def zoom_factory(ax, base_scale=2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
        xdata = event.xdata
        ydata = event.ydata
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
        # force re-draw
        plt.draw()

    # get the figure of interest
    fig = ax.get_figure()
    # attach the call back
    fig.canvas.mpl_connect('scroll_event', zoom_fun)

    # return the function
    return zoom_fun


leds_dic = {}

# Default
CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960

# Defining the dimensions of checkerboard
CHECKERBOARD = (7, 4)
# Termination criteria for refining the detected corners
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
cameraK = np.eye(3).astype(np.float64)
distCoeff = np.zeros((4, 1)).astype(np.float64)

ENABLE = 1
DISABLE = 0

DONE = 'DONE'
NOT_SET = 'NOT_SET'

READ = 0
WRITE = 1

ERROR = -1
SUCCESS = 1

DEBUG = ENABLE

# SENSOR_NAME = "Droidcam"
SENSOR_NAME = "Rift"

EXTERNAL_TOOL_CALIBRATION = 'calibration_json'
RECTIFY_MAP = "improved_params2.xml"
CAM_DELAY = 1
USE_EXTERNAL_TOOL_CALIBRAION = ENABLE
RER_MAX = 100

DO_ESTIMATE_POSE = ENABLE
DO_SOLVEPNP_REFINE = DISABLE
DO_UNDISTORT = DISABLE
DO_SOLVEPNP_RANSAC = ENABLE
USE_PRINT_FRAME = DISABLE
PRINT_FRAME_INFOS = DISABLE

RT_TEST = ENABLE
DYNAMIC_RT = 0
STATIC_RT = 1
DYNAMIC_RT_MED = 2
DYNAMIC_RT_QUIVER = 3
RT_INLIERS = 4
MED_RT = 5

ORIGIN = 'rifts2_left.json'
TARGET = 'rifts2_left.json'
JSON_FILE = 'stereo_json'

MAX_BLOB_SIZE = 250

CV_FINDCONTOUR_LVL = 140
CV_THRESHOLD = 170
CV_MIN_THRESHOLD = 0
CV_MID_THRESHOLD = 30

D3DT = 4
D3D = 3
D2D = 2

LOOP_CNT = 100

cam_json = ['WMTD307H601E9L.json', 'WMTD302J600GA9.json']

cam_info_struct = {'idx': NOT_SET,
                   'port': NOT_SET,
                   'display': {'width': CAP_PROP_FRAME_WIDTH, 'height': CAP_PROP_FRAME_HEIGHT, 'rotate': 0},

                   'blobs': [],
                   'med_blobs': [],

                   'distorted_2d': [],
                   'undistorted_2d': [],

                   'cam_cal': {'cameraK': cameraK, 'dist_coeff': distCoeff},

                   'detect_status': [NOT_SET, 0, 0],

                   'track_cal': {'data': [], 'recording': {'name': NOT_SET}},

                   'D_R_T': {'rvecs': NOT_SET, 'tvecs': NOT_SET},
                   'D_R_T_A': [],
                   'RER': {'C_R_T': {'rvecs': NOT_SET, 'tvecs': NOT_SET}}}
