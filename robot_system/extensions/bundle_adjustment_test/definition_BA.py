import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
import kornia as K
import urllib.request as urllib
import bz2

import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

from robot_system.resource.definition import *
from robot_system.resource.robot_system_data import *
from robot_system.common import *
from robot_system.calibration import *
from led_calibration_BA import *

static_lrt_ap3p = {'rvecs': [1.0408751701777958, -1.3564503295761272, 0.6099095726173165],
                   'tvecs': [0.07094294594784116, -0.018391832686427717, 0.44275118012280207]}

static_rrt_ap3p = {'rvecs': [1.1983530105564564, -1.0109365907303884, 0.5152111789967048],
                   'tvecs': [0.152945783756068, -0.017517379467757346, 0.41892150479975304]}


def triangluate(l_r_t, r_r_t, l_blob, r_blob):
    l_rotation, jacobian = cv2.Rodrigues(l_r_t.rvecs)
    r_rotation, jacobian = cv2.Rodrigues(r_r_t.rvecs)

    l_projection = np.dot(default_cameraK, np.hstack((l_rotation, l_r_t.tvecs)))
    r_projection = np.dot(default_cameraK, np.hstack((r_rotation, r_r_t.tvecs)))

    triangulation = cv2.triangulatePoints(l_projection, r_projection, l_blob, r_blob)
    homog_points = triangulation.transpose()
    return cv2.convertPointsFromHomogeneous(homog_points).ravel()


def remake_3d_points(dump_data):
    group_cnt = len(dump_data[GROUP_DATA_INFO])
    n_points = len(dump_data[LED_INFO])
    n_observations = group_cnt * LED_CNT

    srt_remake_3d = np.empty((n_observations, 3))
    drt_remake_3d = np.empty((n_observations, 3))
    rigid_remake = np.empty((n_observations, 3))
    points_3d = np.empty((n_points, 3))
    for i, leds in enumerate(dump_data[LED_INFO]):
        points_3d[i] = leds.get_pos()

    pos = 0
    for key, value in dump_data[GROUP_DATA_INFO].items():
        for idx, led_data in enumerate(value):
            pair_xy = led_data.get_pair_xy()
            led_pair_cnt = len(pair_xy)
            if led_pair_cnt >= 2:
                if DEBUG > DEBUG_LEVEL.LV_2:
                    print('led num ', idx, ' ', pair_xy)
                comb_led = list(itertools.combinations(pair_xy, 2))
                for comb_data in comb_led:
                    if DEBUG > DEBUG_LEVEL.LV_2:
                        print('comb_led idx: ', idx, ' : ', comb_data)

                    cam_l = comb_data[0]
                    cam_r = comb_data[1]
                    cam_l_id = cam_l['cidx']
                    cam_r_id = cam_r['cidx']
                    l_blob = (float(cam_l['blob'].x), float(cam_l['blob'].y))
                    r_blob = (float(cam_r['blob'].x), float(cam_r['blob'].y))

                    d_l_r_t = dump_data[MEASUREMENT_INFO][cam_l_id].get_r_t().get_curr_r_t()
                    d_r_r_t = dump_data[MEASUREMENT_INFO][cam_r_id].get_r_t().get_curr_r_t()
                    # print('d_l_r_t', d_l_r_t)
                    # print('d_r_r_t', d_r_r_t)
                    s_l_r_t = R_T(
                        np.array([[[static_lrt_ap3p['rvecs'][0]],
                                   [static_lrt_ap3p['rvecs'][1]],
                                   [static_lrt_ap3p['rvecs'][2]]]],
                                 dtype=np.float64),
                        np.array([[static_lrt_ap3p['tvecs'][0]],
                                  [static_lrt_ap3p['tvecs'][1]],
                                  [static_lrt_ap3p['tvecs'][2]]], dtype=np.float64))

                    s_r_r_t = R_T(
                        np.array([[[static_rrt_ap3p['rvecs'][0]],
                                   [static_rrt_ap3p['rvecs'][1]],
                                   [static_rrt_ap3p['rvecs'][2]]]],
                                 dtype=np.float64),
                        np.array([[static_rrt_ap3p['tvecs'][0]],
                                  [static_rrt_ap3p['tvecs'][1]],
                                  [static_rrt_ap3p['tvecs'][2]]], dtype=np.float64))

                    drt_point = triangluate(d_l_r_t, d_r_r_t, l_blob, r_blob)
                    drt_remake_3d[pos] = drt_point
                    srt_point = triangluate(s_l_r_t, s_r_r_t, l_blob, r_blob)
                    srt_remake_3d[pos] = srt_point
                    pos += 1

    pos = 0
    for key, value in dump_data[GROUP_DATA_INFO].items():
        # get led number only remake DONE
        for g_data in value:
            if len(g_data.get_pair_xy()) > 0:
                remake_blob = g_data.get_remake_3d().blob
                rigid_remake[pos] = np.array([remake_blob.x, remake_blob.y, remake_blob.z])
                pos += 1
    return points_3d, drt_remake_3d, srt_remake_3d, rigid_remake


def test_draw(ax, *args):
    pts = args[0][0]
    name = args[0][1]
    c = args[0][2]
    if name == 'points_3d':
        size = 8
        alpha = 0.5
    else:
        size = 8
        alpha = 0.5
    x = [led_data[0] for led_data in pts]
    y = [led_data[1] for led_data in pts]
    z = [led_data[2] for led_data in pts]
    ax.scatter(x, y, z, marker='o', s=size, color=c, alpha=alpha, label=name)
    plt.legend()


def draw_dump_point(dump_data):
    plt.style.use('default')
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111, projection='3d', title='3D_point')

    points_3d, drt_remake, srt_remake, rigid_remake = remake_3d_points(dump_data)

    print('origin\n', points_3d)
    test_draw(ax, [points_3d, 'points_3d', 'blue'])
    # test_draw(ax, [drt_remake, 'drt_remake', 'black'])
    # test_draw(ax, [srt_remake, 'srt_remake', 'black'])
    # test_draw(ax, [rigid_remake, 'rigid_remake', 'green'])

    ax.scatter(0, 0, 0, marker='o', color='purple', s=20)
    ax.set_xlim([-0.7, 0.7])
    ax.set_xlabel('X')
    ax.set_ylim([-0.7, 0.7])
    ax.set_ylabel('Y')
    ax.set_zlim([-0.7, 0.7])
    ax.set_zlabel('Z')
    scale = 1.5
    f = zoom_factory(ax, base_scale=scale)
    return ax


def get_fundamental_matrix(points1, points2):
    """
    Computes the fundamental matrix from corresponding points in two images using the 5 point algorithm.

    Args:
        points1 (np.ndarray): Nx2 array of points in image 1.
        points2 (np.ndarray): Nx2 array of points in image 2.

    Returns:
        np.ndarray: 3x3 fundamental matrix.
    """
    N = points1.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x1, y1 = points1[i, :]
        x2, y2 = points2[i, :]
        A[i, :] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]

    U, S, V = np.linalg.svd(A)
    F = V[-1, :].reshape(3, 3)
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V

    return F


def get_camera_matrices(fundamental_matrix, K1, K2):
    """
    Computes the camera matrices from the fundamental matrix and the intrinsic parameters.

    Args:
        fundamental_matrix (np.ndarray): 3x3 fundamental matrix.
        K1 (np.ndarray): 3x3 intrinsic parameters for camera 1.
        K2 (np.ndarray): 3x3 intrinsic parameters for camera 2.

    Returns:
        tuple: (3x4 projection matrix for camera 1, 3x4 projection matrix for camera 2)
    """
    E = K2.T @ fundamental_matrix @ K1
    U, S, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ V
    R2 = U @ W.T @ V
    t1 = U[:, 2].reshape(3, 1)
    t2 = -t1

    P1 = np.hstack((R1, t1))
    P2 = np.hstack((R2, t2))

    return P1, P2


def decompose_projection_matrix(P):
    K, R = np.linalg.qr(P[:, :3])
    T = np.dot(-R.T, np.linalg.inv(K)) @ P[:, -1]
    return R, T


def camera_displacement(r1, r2, t1, t2):
    # print('r1 ', r1)
    # print('r2 ', r2)
    Rod1, _ = cv2.Rodrigues(r1)
    Rod2, _ = cv2.Rodrigues(r2)
    R1to2 = Rod2.dot(Rod1.T)
    rvec1to2, _ = cv2.Rodrigues(R1to2)
    tvec1to2 = -R1to2.dot(t1) + t2

    # print('Rod1\n', Rod1)
    # print('Rod2\n', Rod2)
    # print('rvec1to2\n', rvec1to2.T)
    # print('tvec1to2\n', tvec1to2.T)

    return rvec1to2, tvec1to2


def inverse_matrix(r12, t12, r1, t1):
    # print('r12\n', r12)
    # print('t12\n', t12)
    # print('r1\n', r1)
    # print('t1\n', t1)
    Rod1, _ = cv2.Rodrigues(r1)
    R1to2, _ = cv2.Rodrigues(r12)
    Rod2 = R1to2.dot(np.linalg.inv(Rod1.T))
    r2, _ = cv2.Rodrigues(Rod2)
    t2 = t12 + R1to2.dot(t1)
    #
    # print('r2 ', r2)
    # print('t2 ', t2)
    return r2, t2.reshape(3, 1)


def add_RT(R1, T1, R2, T2):
    R = np.matmul(R1, R2)
    T = np.matmul(R1, T2) + T1
    return R, T


def TtoA(T):
    A = []
    for i, t in enumerate(T):
        for tt in t:
            A.append(np.array(tt))
    return np.array(A)


def AtoT(A):
    return torch.Tensor(np.array(A))


def print_array_info(A):
    print(f'name:{A.name}', f'len:{len(A.array)}', '\n', A.array)


class NAME:
    def __init__(self, array, name):
        self.array = array
        self.name = name


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * N_CAM_PARAMS].reshape((n_cameras, N_CAM_PARAMS))
    points_3d = params[n_cameras * N_CAM_PARAMS:].reshape((n_points, 3))

    # set code function here
    points_proj = []
    for i, POINT_3D in enumerate(points_3d[point_indices]):
        POINT_2D_PROJ, _ = cv2.projectPoints(POINT_3D,
                                             np.array(camera_params[camera_indices][i][:3]),
                                             np.array(camera_params[camera_indices][i][3:6]),
                                             default_cameraK,
                                             default_distCoeff)
        points_proj.append(POINT_2D_PROJ[0][0])

    points_proj = np.array(points_proj)
    return (abs(points_proj - points_2d)).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * N_CAM_PARAMS + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(N_CAM_PARAMS):
        A[2 * i, camera_indices * N_CAM_PARAMS + s] = 1
        A[2 * i + 1, camera_indices * N_CAM_PARAMS + s] = 1
    for s in range(3):
        A[2 * i, n_cameras * N_CAM_PARAMS + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * N_CAM_PARAMS + point_indices * 3 + s] = 1
    return A
