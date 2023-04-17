import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import subprocess
from operator import itemgetter, attrgetter
import re
import subprocess
import traceback
import math
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, utils
from torch.utils import data
from collections import OrderedDict
from dataclasses import dataclass

from opencv_study.machine_learning.vectors_data import *

ENABLE = 1
DISABLE = 0

DONE = 'DONE'
NOT_SET = 'NOT_SET'

READ = 0
WRITE = 1

ERROR = 'ERROR'
SUCCESS = 'SUCCESS'
CONTINUE = 'CONTINUE'


CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
angle_spec = 70


def read_led_pts(fname):
    pts = []
    with open(f'{fname}.txt', 'r') as F:
        a = F.readlines()
    for idx, x in enumerate(a):
        m = re.match(
            '\{ \.pos *= \{+ *(-*\d+.\d+),(-*\d+.\d+),(-*\d+.\d+) *\}+, \.dir *=\{+ *(-*\d+.\d+),(-*\d+.\d+),(-*\d+.\d+) *\}+, \.pattern=(\d+) },',
            x)
        x = float(m.group(1))
        y = float(m.group(2))
        z = float(m.group(3))
        u = float(m.group(4))
        v = float(m.group(5))
        w = float(m.group(6))
        _pos = list(map(float, [x, y, z]))
        _dir = list(map(float, [u, v, w]))
        pts.append({'idx': idx, 'pos': _pos, 'dir': _dir, 'pair_xy': [], 'remake_3d': [],
                    'min': {'dis': 10, 'remake': []}})
    return pts


def make_camera_array():
    dist_to_controller = 0.5

    depth = 0.5
    step = 3
    dx = 360
    dz = 180

    cam_id = 0
    cam_pose = []

    for k in range(int(dx / step)):
        # for k in range(1):
        delta = math.radians(k * step)  # alpha : x 축과 이루는 각도 (0 ~ 360도)
        for m in range(int(dz / step)):
            theta = math.radians(m * step)  # beta : z 축과 이루는 각도 (0 ~ 180도)
            # theta = math.radians(90)  # beta : z 축과 이루는 각도 (0 ~ 180도)

            x = math.cos(delta) * math.sin(theta) * depth
            y = math.sin(delta) * math.sin(theta) * depth
            z = math.cos(theta) * depth
            # print(np.array([x, y, z]))

            u = -x / depth
            v = -y / depth
            w = -z / depth
            Z3, Z2, Z1 = w, v, u
            if Z3 == -1 or Z3 == 1:
                alpha = 0
            else:
                # alpha = math.degrees(math.acos(np.clip(-Z2 / math.sqrt(1 - Z3*Z3), -1.0, 1.0)))
                alpha = math.degrees(math.acos(np.clip(-Z2 / math.sqrt(1 - Z3 * Z3), -1.0, 1.0)))
            if Z1 < 0:
                alpha = 360 - alpha
            beta = math.degrees(math.acos(Z3))
            # beta = math.degrees(math.acos(Z3))
            gamma = 0

            cam_pose.append({
                'idx': cam_id,
                'position_view': vector3(x, y, z),
                'position': vector3(0.0, 0.0, dist_to_controller),
                'direction': ('zxz', [-alpha, -beta, -gamma]),
                'orient': get_quat_from_euler('zxz', [-alpha, -beta, -gamma])
            })
            cam_id += 1
    # check_projection(cam_pose, cam_array)

    return cam_pose


def get_facing_dot(datas, cam_pose):
    pts_facing = []
    leds_array = []

    for i, data in enumerate(datas):
        led_id = int(data['idx'])
        # origin
        temp = transfer_point(vector3(data['pos'][0], data['pos'][1], data['pos'][2]), cam_pose)
        ori = rotate_point(vector3(data['dir'][0], data['dir'][1], data['dir'][2]), cam_pose)
        _pos_trans = list(map(float, [temp.x, temp.y, temp.z]))
        # 단위 벡터 생성
        normal = nomalize_point(vector3(temp.x, temp.y, temp.z))
        _dir_trans = list(map(float, [ori.x, ori.y, ori.z]))
        # facing dot 찾기/
        ori = nomalize_point(ori)
        facing_dot = get_dot_point(normal, ori)
        rad = np.arccos(np.clip(facing_dot, -1.0, 1.0))
        deg = np.rad2deg(rad)
        angle = math.radians(180.0 - angle_spec)
        rad = np.cos(angle)

        if facing_dot < rad:
            pts_facing.append({'idx': led_id, 'pos': list(map(float, [data['pos'][0], data['pos'][1], data['pos'][2]])),
                               'dir': list(map(float, [data['dir'][0], data['dir'][1], data['dir'][2]])),
                               'pattern': 0})

            leds_array.append(led_id)

    return pts_facing, leds_array



def sliding_window(elements, window_size):
    key_array = []
    key_array_sorted = []
    for i in range(len(elements) - window_size + 1):
        temp = elements[i:i + window_size]
        key_array.append(temp)

    for keys in key_array:
        temp = copy.deepcopy(keys)
        temp.sort()
        key_array_sorted.append(temp)

    return key_array_sorted


from torchvision.transforms import ToTensor
class CustomDataset(Dataset):
    def __init__(self, data_file, transform=None):
        data_set = []
        label_set = []
        for idx, (key, value) in enumerate(data_file.items()):
            # print(key, value)

            for xy in value:
                img_array = np.zeros((CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT), dtype=np.uint8)
                img = Image.fromarray(img_array)
                draw = ImageDraw.Draw(img)
                min_x = CAP_PROP_FRAME_WIDTH
                min_y = CAP_PROP_FRAME_HEIGHT

                for i in range(0, len(xy), 2):
                    x = xy[i:i + 2][0]
                    y = xy[i:i + 2][1]
                    if x < min_x:
                        min_x = x
                    if y < min_y:
                        min_y = y
                    r = 3
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

                crop_img = img.crop((min_x - 200, min_y - 200, min_x + 200, min_y + 200))
                # tensor = ToTensor()(crop_img)
                if transform is not None:
                    image = transform(crop_img)
                # image = image.repeat(32, 1, 1, 1)
                data_set.append(image)
                label_set.append(idx)

        self.data = torch.stack(data_set)
        self.label = torch.Tensor(np.array(label_set)).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.label[index]
        # print('idx', index, image, label)
        return image, label


RESIZE_X = 200
RESIZE_Y = 200


def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 학습 데이터를 DEVICE의 메모리로 보냄
        data, target = data.to(DEVICE), target.to(DEVICE)
        # print(data.shape)  # torch.Size([32, 10])
        # print(target.shape)  # torch.Size([32])
        # print(target)
        optimizer.zero_grad()
        output = model(data)
        # print(output.shape)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            # 모든 오차 더하기
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()

            # 가장 큰 값을 가진 클래스가 모델의 예측입니다.
            # 예측과 정답을 비교하여 일치할 경우 correct에 1을 더합니다.
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


class Net(nn.Module):
    def __init__(self, label):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1*RESIZE_X*RESIZE_Y, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, label)

    def forward(self, x):
        x = x.view(-1, 1*RESIZE_X*RESIZE_Y)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def pickle_data(rw_mode, path, data):
    import pickle
    import gzip
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


if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

    leds = read_led_pts('../rift_6')

    camera_k = np.array([[714.938, 0.0, 676.234],
                         [0.0, 714.938, 495.192],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    dist_coeffs = np.array([[0.074468, ], [-0.024896], [0.005643], [-0.000568]], dtype=np.float64)
    cam_pose = make_camera_array()
    projection_image = {}

    for i, data in enumerate(cam_pose):
        pts_facing, leds_array = get_facing_dot(leds, data)
        cnt = len(pts_facing)
        if cnt < 6:
            continue
        # print(leds_array)
        origin_leds = []
        for led_num in leds_array:
            origin_leds.append([leds[led_num]['pos']])

        origin_leds = np.array(origin_leds, dtype=np.float64)
        obj_cam_pos_n = np.array(data['position'])
        obj_cam_pos_view = np.array(data['position_view'])
        Rod, _ = cv2.Rodrigues(R.from_quat(np.array(data['orient'])).as_matrix())

        ret = cv2.projectPoints(origin_leds, Rod, obj_cam_pos_n,
                                camera_k, dist_coeffs)
        xx, yy = ret[0].reshape(len(origin_leds), 2).transpose()

        if cnt < 6:
            continue
        # keys = sliding_window(leds_array, 6)
        # for key_data in keys:
        key_string = ','.join(str(e) for e in leds_array)
        # print(key_string)
        # print(ret[0].ravel())
        if key_string in projection_image:
            # print('found')
            projection_image[key_string].append(ret[0].ravel())
        else:
            # print('not found')
            projection_image[key_string] = [ret[0].ravel()]

    transform = transforms.Compose([
        transforms.Resize((RESIZE_X, RESIZE_Y)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    file = ''.join(['data_file'])
    data = OrderedDict()
    data = projection_image

    pickle_data(WRITE, file, data)
    #
    EPOCHS = 300
    dataset = CustomDataset(projection_image, transform)
    BATCH_SIZE = 256
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # check images
    # img = utils.make_grid(images, padding=0)
    # npimg = img.numpy()
    # plt.figure(figsize=(10, 7))
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()

    model = Net(len(projection_image)).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    import math
    import time
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optimizer)
        test_loss, test_accuracy = evaluate(model, train_loader)
        #
        print('[{}] Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(
            epoch, test_loss, test_accuracy))
    end = time.time()
    print(f"{end - start:.5f} sec")
    # Save the model to a file
    torch.save(model, 'DNN.pt')

