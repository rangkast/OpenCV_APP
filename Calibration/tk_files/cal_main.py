import re
from tkinter import DISABLED
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import cv2
import math
import numpy as np
from cal_search import *
from cal_def import *

def check_facing_dot(idx, cam_pose, noise):
    pts_facing = []
    pts_cam = []

    if noise == DISABLE:
        target_data = leds_dic['pts']
    else:
        target_data = leds_dic['pts_noise']

    for idx, data in enumerate(target_data):
        # print(idx, data)
        
        # 카메라 pose 기준으로 led 좌표 변환
        temp = transfer_point(vector3(data['pos'][0], data['pos'][1], data['pos'][2]), cam_pose)
        _pos_trans = list(map(float, [temp.x, temp.y, temp.z]))
        # 단위 벡터 생성
        normal = nomalize_point(vector3(temp.x, temp.y, temp.z))        

        # led의 방향 벡터 변환
        ori = rotate_point(vector3(data['dir'][0], data['dir'][1], data['dir'][2]), cam_pose)
        _dir_trans = list(map(float, [ori.x, ori.y, ori.z]))

        # facing dot 찾기/
        facing_dot = get_dot_point(normal, ori)
        angle = math.radians(180.0 - angle_spec)
        rad = np.cos(angle)
        if facing_dot < rad:
            pts_facing.append({'idx': idx, 'pos': list(map(float, [data['pos'][0], data['pos'][1], data['pos'][2]])),
                               'dir': list(map(float, [data['dir'][0], data['dir'][1], data['dir'][2]])),
                               'pattern': data['pattern']})
            pts_cam.append({'idx': idx, 'pos': _pos_trans, 'dir': _dir_trans, 'pattern': data['pattern']})
    return pts_facing, pts_cam
#end check_facing_dot()

def read_led_pts_noise(fname):
    with open(fname, 'r') as F:
        a = F.readlines()
    pts = []
    pts_noise = []
    for idx, x in enumerate(a):
        # print(idx,x)
        m = re.match(
            '\{ \.pos *= \{+ *(-*\d+.\d+),(-*\d+.\d+),(-*\d+.\d+) *\}+, \.dir *=\{+ *(-*\d+.\d+),(-*\d+.\d+),(-*\d+.\d+) *\}+, \.pattern=(0x.+) },',
            x)
        x = float(m.group(1))
        y = float(m.group(2))
        z = float(m.group(3))
        u = float(m.group(4))
        v = float(m.group(5))
        w = float(m.group(6))
        _pos = list(map(float, [x, y, z]))
        _dir = list(map(float, [u, v, w]))
        pts.append({'idx': idx, 'pos': _pos, 'dir': _dir, 'pattern': m.group(7)})
        # insert noise
        if insert_coord_noise_led == ENABLE:
            random_offset = add_random_normal()
            x += float(random_offset[0])
            y += float(random_offset[1])
            z += float(random_offset[2])
        if insert_dir_noise_led == ENABLE:
            random_offset = add_random_normal()
            u += float(random_offset[0])
            v += float(random_offset[1])
            w += float(random_offset[2])
        if insert_coord_noise_led or insert_dir_noise_led == ENABLE:
            _pos = list(map(float, [x, y, z]))
            _dir = list(map(float, [u, v, w]))
        pts_noise.append({'idx': idx, 'pos': _pos, 'dir': _dir, 'pattern': m.group(7)})
    print(f'origin{len(pts)}PointsRead')
    leds_dic['pts'] = pts
    if insert_coord_noise_led or insert_dir_noise_led == ENABLE:
        print(f'noise{len(pts_noise)}PointsRead')
        leds_dic['pts_noise'] = pts_noise
#end read_led_pts_noise()

def cam_projection(idx, var, cam_pose, ax, noise, color):
    # Projection을 위한 Data 만들기
    obj_cam_pos_n = np.array(cam_pose['position'])
    # Rotation을 로드리게스 회전 벡터로 변경

    rotR = R.from_quat(np.array(cam_pose['orient']))
    Rod, _ = cv2.Rodrigues(rotR.as_matrix())

    ret = cv2.projectPoints(var, Rod, obj_cam_pos_n, cameraK, distCoeff)
    xx, yy = ret[0].reshape(len(var), 2).transpose()

    pts = []
    # idx가 다를 수 있을까?
    if noise == ENABLE:
        nidx = [x['idx'] for x in trans_leds_array[idx]['pts_noise_facing']]
    else:
        nidx = [x['idx'] for x in trans_leds_array[idx]['pts_facing']]

    for i in range(len(xx)):
        _pos = list(map(float, [xx[i], yy[i]]))
        pts.append({'idx': nidx[i], 'pos': _pos, 'reserved': 0})

    draw_dots(2, pts, ax, color)

    if noise == DISABLE:
        max_x = 0
        max_y = 0
        if abs(max(xx)) > max_x: max_x = abs(max(xx))
        if abs(min(xx)) > max_x: max_x = abs(min(xx))
        if abs(max(yy)) > max_y: max_y = abs(max(yy))
        if abs(min(yy)) > max_y: max_y = abs(min(yy))
        dimen = max(max_x, max_y)
        dimen *= 1.1
        ax.set_xlim([-dimen, dimen])
        ax.set_ylim([-dimen, dimen])
    return xx, yy
#end cam_projection()

def print_result(idx, ax, rvecs, tvecs, blobs):
    position = np.round_(tvecs.reshape(3), 3).tolist()
    rotation = np.round_(R.from_rotvec(rvecs.reshape(3)).as_quat(), 3).tolist()
    print(f"복원한Position:{np.round_(position, 3)}복원한Rotation:{np.round_(rotation, 3)}")

    observed_pos = tvecs.reshape(3)
    observed_rot = R.from_rotvec(rvecs.reshape(3)).as_quat()
    observed_pose = {'position': vector3(observed_pos[0], observed_pos[1], observed_pos[2]),
                     'orient': quat(observed_rot[0], observed_rot[1], observed_rot[2], observed_rot[3])}

    print(f"Offset:Pos[{np.round_(offset['position'], 3)}]Orient[{np.round_(offset['orient'], 3)}]")


    # 관찰된Pose에알고있던카메라Pose를apply해서offset구하기
    remake_offset = pose_apply_inverse(observed_pose, camera_array[idx])
    print(f"PnP로만든Offset:Pos[{np.round_(remake_offset['position'], 3)}]Orient[{np.round_(remake_offset['orient'], 3)}]")

    new_pts = []
    origin_leds = [[x['pos'][0], x['pos'][1], x['pos'][2]] for x in trans_leds_array[idx]['pts_facing']]

    for lpt in origin_leds:
        pt = vector3(lpt[0], lpt[1], lpt[2])
        temp_pt = transfer_point(pt, remake_offset)
        new_pts.append([temp_pt.x, temp_pt.y, temp_pt.z])

    cam_projection(idx, np.array(new_pts), camera_array[idx], ax, DISABLE, 'yellow')

    print('\n')
#end print_result()

def ransac(idx, leds, ax):
    # 변형 및 재구성

    print('ransac cam:', idx, ':', np.round_(camera_array[idx]['orient'], 3))
    # origin
    origin_leds = [[x['pos'][0], x['pos'][1], x['pos'][2]] for x in leds['pts_facing']]

    #원본을 볼 때
    cam_projection(idx, np.array(origin_leds), camera_array[idx], ax, DISABLE, 'white')

    new_pts = []
    if insert_coord_noise_led or insert_dir_noise_led == ENABLE:
        tmp_leds = [[x['pos'][0], x['pos'][1], x['pos'][2]] for x in leds['pts_noise_facing']]
    else:
        tmp_leds = origin_leds.copy()
    for lpt in tmp_leds:
        pt = vector3(lpt[0], lpt[1], lpt[2])
        if add_pose_offset == ENABLE:
            temp_pt = transfer_point(pt, offset)
        else:
            temp_pt = pt
        new_pts.append([temp_pt.x, temp_pt.y, temp_pt.z])

    # 각 카메라에서 노이즈를 봤을 때 어떻게 될까.
    xx, yy = cam_projection(idx, np.array(new_pts), camera_array[idx], ax, ENABLE, 'red')

    blobs_2d_noise = np.stack((xx, yy), axis=1)

    # check assertion
    check = len(origin_leds)
    print('check:', check, ' blobs:', len(blobs_2d_noise))

    if check != len(blobs_2d_noise):
        print("assertion not equal: ", len(blobs_2d_noise))
        ax.set_title(f'assertion error={len(blobs_2d_noise)}')
        return -1

    if check < 4 or len(blobs_2d_noise) < 4:
        print("assertion < 4: ", check)
        ax.set_title(f'assertion error={check}')
        return -1

    _, rvecs, tvecs, inliers = cv2.solvePnPRansac(np.array(origin_leds), blobs_2d_noise, cameraK, distCoeff)

    print('rvecs:', rvecs)
    print('tvecs:', tvecs)
    
    print_result(idx, ax, rvecs, tvecs, blobs_2d_noise)

# end ransac()


def draw_leds_plot(ax):
    plt.style.use('dark_background')
    fig_cam_facing = plt.figure(figsize=(10, 10))

    if do_sansac == ENABLE:   
        fig_ransac = plt.figure(figsize=(10, 10))
        
    for idx, leds in enumerate(trans_leds_array):
        fig_cam_facing.tight_layout()
        ax_cf = fig_cam_facing.add_subplot(4, round(cam_len / 4), idx + 1)
        ax_cf.set_title(f'origin cam {idx}')
        x, y = cam_projection(idx, np.array([[x['pos'][0], x['pos'][1], x['pos'][2]] for x in leds['pts_facing']]),
                       camera_array[idx], ax_cf, DISABLE, 'white')                     
   

        if insert_coord_noise_led or insert_dir_noise_led == ENABLE:
            nx, ny = cam_projection(idx, np.array([[x['pos'][0], x['pos'][1], x['pos'][2]] for x in leds['pts_noise_facing']]),
                           camera_array[idx], ax_cf, ENABLE, 'red')

            # led number가 같지 않으면 의미 없음
            led_num = np.array([x['idx'] for x in leds['pts_facing']])
            led_2d = []  
            led_2d.append({'idx': led_num, 'x': x, 'y': y, 'nx':nx, 'ny':ny})      

            #ToDo algorithm add   
            check_diff(ax, led_2d, idx)

        if do_sansac == ENABLE:
            fig_ransac.tight_layout() 
            ax_ransac = fig_ransac.add_subplot(4, round(cam_len / 4), idx + 1)
            ax_ransac.set_title(f'pnp cam {idx}')

            ransac(idx, leds, ax_ransac)
#end draw_leds_plot()

def show_leds():
    plt.style.use('default')
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # 원래위치
    draw_dots(3, leds_dic['pts'], ax, 'blue')
    for idx, leds in enumerate(trans_leds_array):
        # facing dot
        if (draw_facing_dot == 1):
            draw_dots(3, leds['pts_facing'], ax, 'red')
        if (draw_camera_point == 1):
            '''
            leds_to_world = []
            for x in leds['pts_cam']:
                new_coord = camtoworld(vector3(x['pos'][0], x['pos'][1], x['pos'][2]), camera_array[idx])
                new_dir = camtoworld(vector3(x['dir'][0], x['dir'][1], x['dir'][2]), camera_array[idx])
                leds_to_world.append({'idx': x['idx'], 'pos': list(map(float, [new_coord.x, new_coord.y, new_coord.z])),
                                      'dir': list(map(float, [new_dir.x, new_dir.y, new_dir.z])),
                                      'pattern': x['pattern']})
            draw_dots(3, leds_to_world, ax, 'green')
            '''
            
    # 원점
    ax.scatter(0, 0, 0, marker='o', color='k', s=20)

    if draw_camera_pos == ENABLE:
        for idx, cam_pose in enumerate(camera_array):
            count = len(trans_leds_array[idx]['pts_facing'])
            new_pt = camtoworld(cam_pose['position'], cam_pose)
            ax.scatter(new_pt.x, new_pt.y, new_pt.z, marker='x', color='blue', s=20)
            label = (f"{cam_pose['idx']} [{count}]")
            ax.text(new_pt.x, new_pt.y, new_pt.z, label, size=10)    

    ax.set_xlim([-0.5, 0.5])
    ax.set_xlabel('X')
    ax.set_ylim([-0.5, 0.5])
    ax.set_ylabel('Y')
    ax.set_zlim([-0.5, 0.5])
    ax.set_zlabel('Z')
    draw_leds_plot(ax)
#end show_leds()


# Main start
if __name__ == "__main__":
# LED data 읽고 처리 부분
    read_led_pts_noise('left_control.txt')

    # Camera 위치 별 facing dot searching
    for idx, cam_pose in enumerate(camera_array):
        pts_facing, pts_cam = check_facing_dot(idx, cam_pose, DISABLE)
        trans_leds_array.append({'pts_facing': pts_facing, 'pts_cam': pts_cam})
        if insert_coord_noise_led or insert_dir_noise_led == ENABLE:
            pts_noise_facing, pts_noise_cam = check_facing_dot(idx, cam_pose, ENABLE)
            trans_leds_array[-1]['pts_noise_facing'] = pts_noise_facing
            trans_leds_array[-1]['pts_noise_cam'] = pts_noise_cam

    show_leds()

    # Plot chart Drawing
    plt.show()
