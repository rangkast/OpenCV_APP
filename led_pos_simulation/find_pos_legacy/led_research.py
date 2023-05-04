import pickle
import gzip
import cv2
import glob
import os
import numpy as np
from enum import Enum, auto
import math

DONE = 'DONE'
NOT_SET = 'NOT_SET'

camera_matrix = [
    # cam 0
    [np.array([[712.623, 0.0, 653.448],
               [0.0, 712.623, 475.572],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)],

    # cam 1
    [np.array([[716.896, 0.0, 668.902],
               [0.0, 716.896, 460.618],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.07542], [-0.026874], [0.006662], [-0.000775]], dtype=np.float64)]
]


class POSE_ESTIMATION_METHOD(Enum):
    # Opencv legacy
    SOLVE_PNP_RANSAC = auto()
    SOLVE_PNP_REFINE_LM = auto()
    SOLVE_PNP_AP3P = auto()
    SOLVE_PNP = auto()
    SOLVE_PNP_RESERVED = auto()


ERROR = 'ERROR'
SUCCESS = 'SUCCESS'
pickle_file = './result.pickle'

with gzip.open(pickle_file, 'rb') as f:
    data = pickle.load(f)

led_data = data['LED_INFO']

for i, leds in enumerate(led_data):
    print(f"{i}, {leds}")


# Default solvePnPRansac
def solvepnp_ransac(*args):
    cam_id = args[0][0]
    points3D = args[0][1]
    points2D = args[0][2]
    camera_k = args[0][3]
    dist_coeff = args[0][4]
    # check assertion
    if len(points3D) != len(points2D):
        print("assertion len is not equal")
        return ERROR, NOT_SET, NOT_SET, NOT_SET

    if len(points2D) < 4:
        print("assertion < 4: ")
        return ERROR, NOT_SET, NOT_SET, NOT_SET

    ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(points3D, points2D,
                                                    camera_k,
                                                    dist_coeff)

    return SUCCESS if ret == True else ERROR, rvecs, tvecs, inliers


# solvePnPRansac + RefineLM
def solvepnp_ransac_refineLM(*args):
    cam_id = args[0][0]
    points3D = args[0][1]
    points2D = args[0][2]
    camera_k = args[0][3]
    dist_coeff = args[0][4]
    ret, rvecs, tvecs, inliers = solvepnp_ransac(points3D, points2D, camera_k, dist_coeff)
    # Do refineLM with inliers
    if ret == SUCCESS:
        if not hasattr(cv2, 'solvePnPRefineLM'):
            print('solvePnPRefineLM requires OpenCV >= 4.1.1, skipping refinement')
        else:
            assert len(inliers) >= 3, 'LM refinement requires at least 3 inlier points'
            # refine r_t vector and maybe changed
            cv2.solvePnPRefineLM(points3D[inliers],
                                 points2D[inliers], camera_k, dist_coeff,
                                 rvecs, tvecs)

    return SUCCESS if ret == True else ERROR, rvecs, tvecs, NOT_SET


# solvePnP_AP3P, 3 or 4 points need
def solvepnp_AP3P(*args):
    cam_id = args[0][0]
    points3D = args[0][1]
    points2D = args[0][2]
    camera_k = args[0][3]
    dist_coeff = args[0][4]

    # check assertion
    if len(points3D) != len(points2D):
        print("assertion len is not equal")
        return ERROR, NOT_SET, NOT_SET, NOT_SET

    if len(points2D) < 3 or len(points2D) > 4:
        print("assertion ", len(points2D))
        return ERROR, NOT_SET, NOT_SET, NOT_SET

    ret, rvecs, tvecs = cv2.solvePnP(points3D, points2D,
                                     camera_k,
                                     dist_coeff,
                                     flags=cv2.SOLVEPNP_AP3P)

    return SUCCESS if ret == True else ERROR, rvecs, tvecs, NOT_SET


SOLVE_PNP_FUNCTION = {
    POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC: solvepnp_ransac,
    POSE_ESTIMATION_METHOD.SOLVE_PNP_REFINE_LM: solvepnp_ransac_refineLM,
    POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P: solvepnp_AP3P,
}


def detect_led_lights(image, threshold=240, padding=5):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    # 이진화된 이미지에서 윤곽선을 찾음
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blob_info = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # 주변 부분을 포함하기 위해 패딩을 적용
        x -= padding
        y -= padding
        w += padding * 2
        h += padding * 2
        # 박스 그리기
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)
        cv2.putText(image, str(idx), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), lineType=cv2.LINE_AA)

        blob_info.append([x, y, w, h])

    return image, blob_info


def load_images(path):
    image_files = glob.glob(os.path.join(path, "*.png"))
    images = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY) for img in image_files]
    return images, image_files


def find_center(frame, SPEC_AREA):
    x_sum = 0
    t_sum = 0
    y_sum = 0
    g_c_x = 0
    g_c_y = 0
    m_count = 0

    (X, Y, W, H) = SPEC_AREA

    for y in range(Y, Y + H):
        for x in range(X, X + W):
            x_sum += x * frame[y][x]
            t_sum += frame[y][x]
            m_count += 1

    for x in range(X, X + W):
        for y in range(Y, Y + H):
            y_sum += y * frame[y][x]

    if t_sum != 0:
        g_c_x = x_sum / t_sum
        g_c_y = y_sum / t_sum

    if g_c_x == 0 or g_c_y == 0:
        return 0, 0
    #
    # result_data_str = ' x ' + f'{g_c_x}' + ' y ' + f'{g_c_y}'
    # print(result_data_str)

    return g_c_x, g_c_y

def rotation_matrix_to_euler_angles_degrees(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    # 각도 단위를 라디안에서 도로 변환
    x = np.degrees(x)
    y = np.degrees(y)
    z = np.degrees(z)

    return np.array([x, y, z])


blob_info = [['CAMERA_0_x90_y0_z90_20230504_150458.png', [5, 14, 6, 12, 7, 8], [1, 3, -1, 2, -1, 0]]]


def display_images(images, image_files):
    index = 0
    while True:
        blob_img, blob_area = detect_led_lights(images[index], 100, 5)
        # 파일 이름을 이미지 상단에 표시
        file_name = os.path.basename(image_files[index])
        cv2.putText(blob_img, file_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Image Viewer", blob_img)
        METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P

        for blobs in blob_info:
            if blobs[0] == file_name:
                print('START TEST')
                image_points = []
                object_points = []
                for idx, area in enumerate(blob_area):
                    cx, cy = find_center(blob_img, (area[0], area[1], area[2], area[3]))
                    image_points.append([cx, cy])
                    object_points.append(led_data[blobs[1][idx]])

                points2D = [None] * (len(blobs[2]) - blobs[2].count(-1))
                for idx, val in enumerate(blobs[2]):
                    if val != -1:
                        points2D[val] = image_points[idx]

                points3D = [None] * (len(blobs[2]) - blobs[2].count(-1))
                for idx, val in enumerate(blobs[2]):
                    if val != -1:
                        points3D[val] = object_points[idx]

                points2D = np.array(points2D, dtype=np.float64)
                points3D = np.array(points3D, dtype=np.float64)
                print('point_3d\n', points3D)
                print('point_2d\n', points2D)

                parsed_array = file_name.split('_')
                cam_id = int(parsed_array[1])
                camera_k = camera_matrix[cam_id][0]
                dist_coeff = camera_matrix[cam_id][1]
                INPUT_ARRAY = [
                    cam_id,
                    points3D,
                    points2D,
                    camera_k,
                    dist_coeff
                ]

                ret, rvec, tvec, inliers = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)

                print('rvecs\n', rvec)
                print('tvecs\n', tvec)

                # Extract rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                R_inv = np.linalg.inv(R)
                # Camera position (X, Y, Z)
                Cam_pos = -R_inv @ tvec
                X, Y, Z = Cam_pos.ravel()
                unit_z = np.array([0, 0, 1])

                Zc = np.reshape(unit_z, (3, 1))
                Zw = np.dot(R_inv, Zc)  # world coordinate of optical axis
                zw = Zw.ravel()

                pan = np.arctan2(zw[1], zw[0]) - np.pi / 2
                tilt = np.arctan2(zw[2], np.sqrt(zw[0] * zw[0] + zw[1] * zw[1]))

                # roll
                unit_x = np.array([1, 0, 0])
                Xc = np.reshape(unit_x, (3, 1))
                Xw = np.dot(R_inv, Xc)  # world coordinate of camera X axis
                xw = Xw.ravel()
                xpan = np.array([np.cos(pan), np.sin(pan), 0])

                roll = np.arccos(np.dot(xw, xpan))  # inner product
                if xw[2] < 0:
                    roll = -roll

                roll = math.degrees(roll)
                pan = math.degrees(pan)
                tilt = math.degrees(tilt)

                print('world coord info')
                print('pos', X, Y, Z)
                print('degrees', 'roll', roll, 'pan', pan, 'tilt', tilt)
                print('camera coord info')

                R_blender = np.array([
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                ])
                tvec_blender = np.matmul(R_blender, tvec)
                R_blender = np.matmul(R_blender, np.matmul(R, R_blender.T))
                euler_angles_degrees = rotation_matrix_to_euler_angles_degrees(R_blender)

                print("Blender Camera Location:", tvec_blender)
                print("Blender Camera Euler angles (degrees):", euler_angles_degrees)

        key = cv2.waitKey(0)
        if key == ord('n'):
            index += 1
            if index >= len(images):
                print('end of files')
                break
        elif key & 0xFF == 27:
            print('ESC pressed')
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = '/home/rangkast.jeong/render_output/'
    images, image_files = load_images(image_path)
    display_images(images, image_files)
