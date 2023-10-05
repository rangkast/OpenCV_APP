import subprocess
import re
from Advanced_Function import *

# left: [0.1, 0.2, 0.3] | [1.0, 1.1, 1.2]
# right: [0.4, 0.5, 0.6] | [1.3, 1.4, 1.5]


def get_pose_from_adb_log():
    cmd = ["adb", "logcat"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    pattern = re.compile(r"POSE : (\d+\.\d+) (\d+\.\d+) (\d+\.\d+) (\d+\.\d+)")

    while True:
        line = process.stdout.readline().decode('utf-8')
        if line == '' and process.poll() is not None:
            break
        match = pattern.search(line)
        if match:
            w, x, y, z = map(float, match.groups())
            yield w, x, y, z

import numpy as np
import cv2
import matplotlib.pyplot as plt

def parse_log(log_file):
    with open(log_file, 'r') as file:
        for line in file:
            parts = line.strip().split(":")
            side = parts[0].strip()
            data = parts[1].strip()
            rvec_str, tvec_str = data.split('|')
            
            rvec = np.array([float(x) for x in rvec_str.strip('[]').split(',')])
            tvec = np.array([float(x) for x in tvec_str.strip('[]').split(',')])
            
            yield side, rvec, tvec

def get_location_direction(log_file):
    for side, rvec, tvec in parse_log(log_file):
        location, direction = world_location_rotation_from_opencv(rvec, tvec)
        yield side, location, direction

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for side, location, direction in get_location_direction('test.txt'):
    # location을 시작점으로, direction을 화살표 방향으로 하는 quiver를 그림
    ax.quiver(location[0], location[1], location[2], direction[0], direction[1], direction[2], color='b' if side == 'left' else 'r')

    # 적절한 축 범위 설정 (필요에 따라 수정)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
    
    plt.draw()
    plt.pause(0.5)
    ax.cla()

plt.show()
