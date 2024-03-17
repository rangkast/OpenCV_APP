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

import numpy as np

# 초기 원의 중심을 무작위로 배치하는 함수
def initialize_centers(plane_coords, LED_POINTS):
    indices = np.random.choice(plane_coords.shape[0], size=LED_POINTS, replace=False)
    return plane_coords[indices]

# 모든 원의 중심 간의 최소 거리를 계산하는 함수
def compute_min_distances(centers):
    distances = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2))
    np.fill_diagonal(distances, np.inf)  # 자기 자신과의 거리는 무한대로 설정
    return np.min(distances, axis=1)
def adjust_centers_constrained(centers, plane_coords, adjustment_step=0.01, distance_threshold=0.02):
    adjusted_centers = np.copy(centers)
    for i, center in enumerate(centers):
        direction = np.zeros(center.shape)
        for j, other_center in enumerate(centers):
            if i != j:
                dir_vector = center - other_center
                direction += dir_vector / np.linalg.norm(dir_vector)
        direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) != 0 else direction
        new_center = center + direction * adjustment_step
        # 영역 내 검사: 새 중심이 plane_coords 영역 내에 있는지 확인
        closest_dist_to_plane = np.min(np.sqrt(np.sum((plane_coords - new_center)**2, axis=1)))
        if closest_dist_to_plane < distance_threshold:
            adjusted_centers[i] = new_center
    return adjusted_centers

# 새로운 최적화 함수 정의
def optimize_centers_constrained(plane_coords, LED_POINTS, num_iterations=100, adjustment_step=0.01, distance_threshold=0.02):
    centers = initialize_centers(plane_coords, LED_POINTS)
    for _ in range(num_iterations):
        centers = adjust_centers_constrained(centers, plane_coords, adjustment_step, distance_threshold)
    return centers


# 예시 데이터
num_points = 1000
R = 0.3
radius_outer = 0.02
radius_inner = 0.015
LED_POINTS = 15

theta = np.linspace(0, 2 * np.pi, num_points)
phi = np.linspace(0, np.pi, num_points)
theta, phi = np.meshgrid(theta, phi)

x = R * np.sin(phi) * np.cos(theta)
y = R * np.sin(phi) * np.sin(theta)
z = R * np.cos(phi)

outer_cut_condition = np.sqrt(x**2 + y**2 + (z - R)**2) <= radius_outer
inner_cut_condition = np.sqrt(x**2 + y**2 + (z - R)**2) >= radius_inner
cut_condition = np.logical_and(outer_cut_condition, inner_cut_condition)
plane_coords = np.vstack([x[cut_condition].ravel(), y[cut_condition].ravel(), z[cut_condition].ravel()]).T

# 최적화 실행
# 제약 조건을 적용한 최적화 실행
# optimized_centers_constrained = optimize_centers_constrained(plane_coords, LED_POINTS, num_iterations=100, adjustment_step=0.01, distance_threshold=0.02)



from sklearn.cluster import KMeans

def apply_kmeans_clustering(coords, num_leds):
    """
    K-means 클러스터링을 적용하여 면을 구성하는 점들 중에서
    num_leds 개수만큼의 대표 점을 선택한다.

    Parameters:
    - coords: 면을 구성하는 점들의 좌표 배열 (numpy array)
    - num_leds: 선택하고자 하는 LED(점)의 개수

    Returns:
    - centers: 선택된 점들의 좌표 (numpy array)
    """
    kmeans = KMeans(n_clusters=num_leds, random_state=0).fit(coords)
    centers = kmeans.cluster_centers_
    return centers


# K-means 클러스터링 적용
selected_centers = apply_kmeans_clustering(plane_coords, LED_POINTS)


# 잘린 부분의 표면 점들 시각화
ax1.scatter(plane_coords[:,0], plane_coords[:,1], plane_coords[:,2], color='gray', marker='.', alpha=1)
# 잘린 부분의 표면 점들 시각화
ax1.scatter(selected_centers[:,0], selected_centers[:,1], selected_centers[:,2], color='red', marker='.', alpha=1, s=5)

          
# show results
plt.show()
