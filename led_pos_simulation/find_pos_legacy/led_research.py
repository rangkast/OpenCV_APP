import pickle
import gzip
import cv2
import glob
import os
import numpy as np
from enum import Enum, auto
import math
import platform
from scipy.spatial.transform import Rotation as Rot

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
default_dist_coeffs = np.zeros((4, 1))


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
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
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
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(image, str(idx), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), lineType=cv2.LINE_AA)

        blob_info.append([x, y, w, h])

    return image, binary_image, blob_info


def load_data(path):
    image_files = glob.glob(os.path.join(path, "*.png"))
    data_files = glob.glob(os.path.join(path, "*.txt"))
    images = [cv2.imread(img) for img in image_files]
    return images, image_files, data_files


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
    result_data_str = ' x ' + f'{g_c_x}' + ' y ' + f'{g_c_y}'
    print(result_data_str)

    return g_c_x, g_c_y


blob_info = [
    ['CAMERA_0_x90_y0_z90', [5, 14, 6, 12, 7, 8], [1, 3, -1, 2, -1, 0]],
    ['CAMERA_0_x90_y15_z90', [5, 12, 8, 14, 6, 7], [1, 2, 0, 3, -1, -1]],
    ['CAMERA_0_x74_y15_z84', [5, 12, 8, 14, 6, 7], [1, 2, 0, 3, -1, -1]]

]
def display_images(images, image_files, data_files):
    index = 0
    print('data_files\n', data_files)
    print('image_files\n', image_files)
    while True:
        # draw_img, blob_img, blob_area = detect_led_lights(images[index], 100, 5)
        # 파일 이름을 이미지 상단에 표시
        img_name = os.path.basename(image_files[index])
        key = cv2.waitKey(0)

        if key == ord('n'):
            index += 1
            if index >= len(images):
                print('end of files')
                break
        elif key == ord('c'):
            METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC
            draw_img = images[index].copy()
            draw_img, blob_img, blob_area = detect_led_lights(draw_img, 100, 5)
            cv2.putText(draw_img, img_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            for blobs in blob_info:
                if blobs[0] in img_name:
                    print('START TEST')
                    image_points = []
                    object_points = []
                    for idx, area in enumerate(blob_area):
                        cx, cy = find_center(blob_img, (area[0], area[1], area[2], area[3]))
                        cv2.circle(draw_img, (int(cx), int(cy)), 1, (0, 0, 0), -1)
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

                    parsed_array = img_name.split('_')
                    cam_id = int(parsed_array[1])
                    camera_k = camera_matrix[cam_id][0]

                    # blender에서 왜곡계수 처리방법 확인중
                    dist_coeff = default_dist_coeffs

                    INPUT_ARRAY = [
                        cam_id,
                        points3D,
                        points2D,
                        camera_k,
                        dist_coeff
                    ]

                    ret, rvec, tvec, inliers = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)

                    print('RT from OpenCV SolvePnP')
                    print('rvecs\n', rvec.ravel())
                    print('tvecs\n', tvec.ravel())

                    # 3D 점들을 2D 이미지 평면에 투영
                    image_points, _ = cv2.projectPoints(points3D, rvec, tvec, camera_k, dist_coeff)
                    image_points = np.squeeze(image_points)

                    # 투영된 2D 이미지 점 출력
                    print("Projected 2D image points:")
                    print(image_points)

                    # 빨간색점은 opencv pose-estimation 투영 결과
                    for repr_blob in image_points:
                        cv2.circle(draw_img, (int(repr_blob[0]), int(repr_blob[1])), 1, (0, 0, 255), -1)

                    # data load from add on
                    for data_path in data_files:
                        if blobs[0] in data_path:
                            projectionMatrix = np.loadtxt(data_path)
                            intrinsic, rotationMatrix, homogeneousTranslationVector = cv2.decomposeProjectionMatrix(
                                projectionMatrix)[:3]
                            camT = -cv2.convertPointsFromHomogeneous(homogeneousTranslationVector.T)
                            camR = Rot.from_matrix(rotationMatrix)
                            blender_tvec = camR.apply(camT.ravel())
                            blender_rvec = camR.as_rotvec()
                            print('RT from Blender to Opencv')
                            print('rvecs\n', blender_rvec)
                            print('tvecs\n', blender_tvec)
                            blender_image_points, _ = cv2.projectPoints(points3D, blender_rvec, blender_tvec, camera_k,
                                                                        dist_coeff)
                            blender_image_points = np.squeeze(blender_image_points)
                            print("Projected 2D image points:")
                            print(blender_image_points)
                            for repr_blob in blender_image_points:
                                cv2.circle(draw_img, (int(repr_blob[0]), int(repr_blob[1])), 1, (255, 0, 0), -1)

                    cv2.namedWindow("Processed Image")
                    cv2.imshow("Processed Image", draw_img)

        elif key & 0xFF == 27:
            print('ESC pressed')
            break

        cv2.namedWindow("Image")
        cv2.imshow("Image", images[index])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    os_name = platform.system()
    image_path = '../blender_3d/render_output'

    images, image_files, data_files = load_data(image_path)
    display_images(images, image_files, data_files)
