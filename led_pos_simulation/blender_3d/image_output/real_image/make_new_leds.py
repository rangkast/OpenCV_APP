import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import TextBox
import cv2
import matplotlib as mpl
import tkinter as tk
from collections import OrderedDict
from dataclasses import dataclass
import pickle
import gzip
import cv2
import glob
import os
from enum import Enum, auto
import math
import platform
from scipy.spatial.transform import Rotation as Rot
import json
import matplotlib.ticker as ticker


READ = 0
WRITE = 1

ERROR = 'ERROR'
SUCCESS = 'SUCCESS'

DONE = 'DONE'
NOT_SET = 'NOT_SET'

def pickle_data(rw_mode, path, data):
    import pickle
    import gzip
    print('path', path)
    try:
        if rw_mode == READ:
            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)
            return data
        elif rw_mode == WRITE:
            with gzip.open(path, 'wb') as f:
                pickle.dump(data, f)
        else:
            print('not support mode')
    except:
        print('file r/w error')
        return ERROR


class Vector:
    def __init__(self, *coords):
        if len(coords) == 1 and isinstance(coords[0], (list, tuple, np.ndarray)):
            coords = coords[0]
        self.coords = np.array(coords)
        self.x = coords[0] if len(coords) > 0 else 0
        self.y = coords[1] if len(coords) > 1 else 0
        self.z = coords[2] if len(coords) > 2 else 0

    def __repr__(self):
        return f"Vector({', '.join(map(str, self.coords))})"

    def __getitem__(self, index):
        return self.coords[index]

    def normalized(self):
        norm = np.linalg.norm(self.coords)
        return Vector(*(self.coords / norm))

def zoom_factory(ax, base_scale=2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_zlim = ax.get_zlim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
        cur_zrange = (cur_zlim[1] - cur_zlim[0]) * .5
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
        ax.set_zlim([(xdata + ydata) / 2 - cur_zrange * scale_factor,
                     (xdata + ydata) / 2 + cur_zrange * scale_factor])
        # force re-draw
        plt.draw()

    # get the figure of interest
    fig = ax.get_figure()
    # attach the call back
    fig.canvas.mpl_connect('scroll_event', zoom_fun)

    # return the function
    return zoom_fun

root = tk.Tk()
width_px = root.winfo_screenwidth()
height_px = root.winfo_screenheight()

# 모니터 해상도에 맞게 조절
mpl.rcParams['figure.dpi'] = 120  # DPI 설정
monitor_width_inches = width_px / mpl.rcParams['figure.dpi']  # 모니터 너비를 인치 단위로 변환
monitor_height_inches = height_px / mpl.rcParams['figure.dpi']  # 모니터 높이를 인치 단위로 변환

# Figure 크기를 모니터 해상도에 맞게 조절하고 제목 추가
fig = plt.figure(figsize=(monitor_width_inches, monitor_height_inches), num='LED Position FinDer')
plt.rcParams.update({'font.size': 7})
# Single Axes 생성
ax1 = fig.add_subplot(111, projection='3d')
ax1.set_title('3D plot')    
ax1.scatter(0, 0, 0, marker='o', color='k', s=20)
ax1.set_xlim([-1.0, 1.0])
ax1.set_ylim([-1.0, 1.0])
ax1.set_zlim([-1.0, 1.0])
scale = 1.5
f = zoom_factory(ax1, base_scale=scale)



def select_points(coords, num_leds, min_start_distance=0):
    if len(coords) == 0:  # coords 배열이 비어 있으면 빈 리스트 반환
        return []

    coords = np.array(coords)    
    selected_indices = [0]

    for _ in range(1, num_leds):
        max_dist = 0
        next_index = None

        for i, point in enumerate(coords):
            if i in selected_indices:
                continue
            min_dist_to_selected = min([np.linalg.norm(point - coords[selected_idx]) for selected_idx in selected_indices])

            if min_dist_to_selected >= min_start_distance and min_dist_to_selected > max_dist:
                max_dist = min_dist_to_selected
                next_index = i

        if next_index is not None:
            selected_indices.append(next_index)
        else:
            break


    return [coords[i] for i in selected_indices]


def calculate_all_pairwise_distances(coords):
    num_points = len(coords)
    distances = []

    for i in range(num_points):
        for j in range(i + 1, num_points):  # 중복 계산을 피하기 위해 j는 i+1부터 시작
            dist = np.linalg.norm(coords[i] - coords[j]) + 0.002 * 2
            distances.append(dist)

    return distances

def sequential_closest_distances(coords, led_size):
    visited_indices = [0]
    current_idx = 0
    distances = []

    for _ in range(len(coords) - 1):
        min_dist = float("inf")
        closest_idx = None

        for idx, coord in enumerate(coords):
            if idx not in visited_indices:
                dist = np.linalg.norm(coords[current_idx] - coord) + led_size * 2

                if dist < min_dist:
                    min_dist = dist
                    closest_idx = idx

        visited_indices.append(closest_idx)
        distances.append(min_dist)
        current_idx = closest_idx

    return distances

def draw_sequential_closest_lines(ax, led_coords_o):
    visited_indices = [0]
    current_idx = 0

    while len(visited_indices) < len(led_coords_o):
        min_dist = float("inf")
        closest_idx = None

        for idx, coord in enumerate(led_coords_o):
            if idx not in visited_indices:
                dist = np.linalg.norm(led_coords_o[current_idx] - coord)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = idx

        if closest_idx is not None:
            visited_indices.append(closest_idx)
            ax.plot(
                [led_coords_o[current_idx][0], led_coords_o[closest_idx][0]],
                [led_coords_o[current_idx][1], led_coords_o[closest_idx][1]],
                [led_coords_o[current_idx][2], led_coords_o[closest_idx][2]],
                linewidth=0.5,
                alpha=0.5,
                color='black',
                linestyle='--'
            )
            current_idx = closest_idx
        else:
            break

def plot_diagonal_cut_sphere_with_flat_cut_surface(*args):
     ax1 = args[0][0]
     R = args[0][1]  # 구의 반지름 (상수)
     num_points = args[0][2]
     leds_distance = args[0][3]  # LED 사이 최소 거리
     radius_outer = args[0][4]  # 꼭지점에서의 잘라낼 외부 거리
     radius_inner = args[0][5]  # 꼭지점에서의 잘라낼 내부 거리
     LED_POINTS = args[0][6]
     led_size = args[0][7]
     padding = args[0][8]  # 패딩 영역의 두께 설정
     loop_cnt = 0

     last_valid_R = R
     last_valid_led_coords = []
     last_valid_distance = []
     last_combined_coords = []

     theta = np.linspace(0, 2 * np.pi, num_points)
     phi = np.linspace(0, np.pi, num_points)
     theta, phi = np.meshgrid(theta, phi)

     # make R sphere
     x = R * np.sin(phi) * np.cos(theta)
     y = R * np.sin(phi) * np.sin(theta)
     z = R * np.cos(phi)

     while radius_outer > radius_inner:
          # 꼭지점에서 R 거리 내에 있는 점들만 선택
          outer_cut_condition = np.sqrt(x**2 + y**2 + (z - R)**2) <= radius_outer
          # 꼭지점에서 radius_inner 거리보다 멀리 있는 점들만 선택
          inner_cut_condition = np.sqrt(x**2 + y**2 + (z - R)**2) >= radius_inner
          # 두 조건을 모두 만족하는 점들을 선택
          cut_condition = np.logical_and(outer_cut_condition, inner_cut_condition)
          # 잘린 부분의 표면 점들만 선택
          plane_coords = np.vstack([x[cut_condition].ravel(), y[cut_condition].ravel(),  z[cut_condition].ravel()]).T
          # 패딩 영역 좌표 선택
          outer_padding_mask = np.logical_and(np.sqrt(x**2 + y**2 + (z - R)**2) <= (radius_outer + padding), np.sqrt(x**2 + y**2 + (z - R)**2) > radius_outer)
          inner_padding_mask = np.logical_and(np.sqrt(x**2 + y**2 + (z - R)**2) >= (radius_inner - padding), np.sqrt(x**2 + y**2 + (z - R)**2) < radius_inner)
          total_padding_mask = np.logical_or(outer_padding_mask, inner_padding_mask)
          padding_coords = np.vstack([x[total_padding_mask].ravel(), y[total_padding_mask].ravel(), z[total_padding_mask].ravel()]).T
          # 원래 좌표들과 패딩 좌표들을 결합
          combined_coords = np.vstack((plane_coords, padding_coords))

          # LED 점 선택
          led_coords = select_points(plane_coords, LED_POINTS)[1:]

          if len(led_coords) > 1:  # 유효한 LED 좌표가 있는 경우에만 처리
               distances = sequential_closest_distances(led_coords, led_size)
               print("###### loopcnt ######### ", loop_cnt)
               print("min ", distances)
               print("min ", min(distances))

               # 조건을 만족하는지 확인
               if len(distances) > 0 and min(distances) > leds_distance:
                    # 조건을 만족하면 값 업데이트
                    last_valid_R = radius_outer
                    last_valid_led_coords = led_coords
                    last_valid_distance = distances
                    last_combined_coords = combined_coords

               radius_outer -= 0.01  # R을 줄이고 계속 진행
          else:
               print("EOF")
               break  # 유효한 LED 좌표가 없으면 루프 종료

          loop_cnt += 1


     # 잘린 부분의 표면 점들 시각화
     ax1.scatter(last_combined_coords[:,0], last_combined_coords[:,1], last_combined_coords[:,2], color='gray', marker='.', alpha=0.1)

     # 시각화 부분 수정
     for points in last_valid_led_coords:
          ax1.scatter(points[0], points[1], points[2], color='black', marker='o', s=5)

     draw_sequential_closest_lines(ax1, last_valid_led_coords)

     # 정렬된 distances 출력
     print("Distances in descending order:")
     for dist in sorted(last_valid_distance, reverse=True):
          print(dist)

     print("Final R: ", last_valid_R)

     return last_combined_coords, last_valid_led_coords
    
# 단위 meter
num_points= 1000
R = 0.3
radius_outer = 0.02
radius_inner = 0.015

LED_POINTS = 18
led_distance =  0.01
led_size = 0.002

random_distance = 0.002

padding = 0.005

combined_coords, ret_coords = plot_diagonal_cut_sphere_with_flat_cut_surface([ax1, R, num_points, led_distance, radius_outer, radius_inner, LED_POINTS, led_size, padding])

# Save result data to pickle file
def calculate_normalized_direction(coord):
    # 원점에서 해당 점까지의 벡터를 계산
    vector = np.array(coord)
    # 벡터의 크기(길이)를 계산
    magnitude = np.linalg.norm(vector)
    # 벡터를 정규화하여 단위 벡터로 만듦
    if magnitude == 0:
        return vector  # 원점인 경우 그대로 반환
    else:
        return vector / magnitude

file = './result_new_object.pickle'
data = OrderedDict()

p_coords_changed = []

for i, coord in enumerate(ret_coords):
    # LED 오브젝트의 위치를 조정합니다.
    normalized_direction = Vector(coord).normalized()
    distance_to_o = led_size * 3 / 4
    p_coords_changed.append([round(coord[0] - distance_to_o * normalized_direction[0], 9),
                             round(coord[1] - distance_to_o * normalized_direction[1], 9),
                             round(coord[2], 8)])
import random

def find_random_point_within_distance(combined_coords, ret_coords, led_distance):
    selected_points = []
    
    # combined_coords와 ret_coords를 NumPy 배열로 변환
    combined_coords_array = np.array(combined_coords)
    ret_coords_array = np.array(ret_coords)
    
    # ret_coords의 각 점에 대해 처리
    for point in ret_coords_array:
        # combined_coords 내의 모든 점과의 거리 계산
        distances = np.linalg.norm(combined_coords_array - point, axis=1)
        
        # led_distance 이내의 점들의 인덱스를 찾음
        close_points_indices = np.where(distances <= led_distance)[0]
        
        # 해당하는 점이 있을 경우 랜덤하게 하나 선택
        if len(close_points_indices) > 0:
            selected_index = random.choice(close_points_indices)
            selected_point = combined_coords_array[selected_index]
            selected_points.append(selected_point.tolist())
        else:
            # 해당 거리 내에 점이 없는 경우 None 추가 (또는 다른 처리 방식 선택)
            selected_points.append(None)
    
    return selected_points


def plot_points_within_distance(ax, combined_coords, ret_coords, distance):
    """
    combined_coords에서 각 ret_coords 점에 대해 지정된 거리 내의 모든 점을 찾아 그립니다.
    """
    for point in ret_coords:
        # 각 ret_coords 점에 대해 combined_coords 내 모든 점과의 거리 계산
        distances = np.linalg.norm(combined_coords - point, axis=1)
        # 지정된 거리 이내의 점들의 인덱스를 찾음
        within_distance_indices = np.where(distances <= distance)[0]
        
        # 거리 내의 점들을 초록색으로 그림
        for idx in within_distance_indices:
            ax.scatter(*combined_coords[idx], color='green', s=10)




# 각 점에 대한 방향 벡터를 계산
direction_info = [calculate_normalized_direction(coord) for coord in p_coords_changed]
led_num = 0
for coord, direction in zip(p_coords_changed, direction_info):
    ax1.quiver(coord[0], coord[1], coord[2],  # 화살표의 시작점(각 LED 점의 위치)
               direction[0], direction[1], direction[2],  # 방향
               length=0.05,  # 화살표의 길이 (적절한 값으로 조정)
               color='r',  # 화살표 색상
               alpha=0.6,  # 화살표 투명도
               normalize=True)  # 방향 벡터 정규화벡터 정규화
    
    print(f'{led_num} pos {coord} dir {direction}')    
    led_num += 1

# combined_coords와 p_coords_changed 사용하여 지정된 거리 내의 점들을 그리는 함수 호출
plot_points_within_distance(ax1, combined_coords, p_coords_changed, random_distance)
# 함수 호출
selected_points = find_random_point_within_distance(combined_coords, p_coords_changed, random_distance)
direction_info = [calculate_normalized_direction(coord) for coord in selected_points]
for coord, direction in zip(selected_points, direction_info):
    ax1.quiver(coord[0], coord[1], coord[2],  # 화살표의 시작점(각 LED 점의 위치)
               direction[0], direction[1], direction[2],  # 방향
               length=0.05,  # 화살표의 길이 (적절한 값으로 조정)
               color='black',  # 화살표 색상
               alpha=0.6,  # 화살표 투명도
               normalize=True)  # 방향 벡터 정규화벡터 정규화
    ax1.scatter(coord[0], coord[1], coord[2], color='black', marker='o', s=5)

print('selected_points')
print(selected_points)

data['LED_INFO'] = p_coords_changed
data['MODEL_INFO'] = combined_coords
data['DIRECTION_INFO'] = direction_info

ret = pickle_data(WRITE, file, data)
if ret != ERROR:
    print('data saved')


# show results
plt.show()
