import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import TextBox
import cv2
import matplotlib as mpl
import tkinter as tk
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
CV_MIN_THRESHOLD = 100
CV_MAX_THRESHOLD = 255


def detect_led_lights(image, padding=5):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blob_info = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # 주변 부분을 포함하기 위해 패딩을 적용
        x -= padding
        y -= padding
        w += padding * 2
        h += padding * 2
        blob_info.append([x, y, w, h])

    return blob_info


def blend_images(ax, name, image1, image2, alpha=0.5):
    # 이미지를 읽고, 동일한 크기로 조정합니다.
    _, img1 = cv2.threshold(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                            cv2.THRESH_TOZERO)
    _, img2 = cv2.threshold(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), CV_MIN_THRESHOLD, CV_MAX_THRESHOLD,
                            cv2.THRESH_TOZERO)

    img_draw_1, ax_img_1 = draw_elllipse(ax, 'BL', img1)
    img_draw_2, ax_img_2 = draw_elllipse(ax, 'RE', img2)

    img_draw_blended = cv2.addWeighted(img_draw_1, alpha, img_draw_2, 1 - alpha, 0)
    ax_blended = cv2.addWeighted(ax_img_1, alpha, ax_img_2, 1 - alpha, 0)

    ax.set_title(f'2D Image and Projection of GreySum Blender Obj Util CAM_{name}')
    ax.imshow(ax_blended, cmap='gray')

    cv2.imwrite(name, img_draw_blended)
    return img_draw_blended


def draw_elllipse(ax, name, image):
    # 컨투어 찾기
    img_draw = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    blob_area = detect_led_lights(image, 5)
    for idx, area in enumerate(blob_area):
        (x, y, w, h) = area
        # 무게 중심 계산
        roi = image[y:y + h, x:x + w]
        # 컨투어 찾기
        contours, _ = cv2.findContours(roi.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) > 5:
                ellipse = cv2.fitEllipse(cnt)
                ellipse = ((ellipse[0][0] + x, ellipse[0][1] + y), ellipse[1], ellipse[2])  # 타원의 중심 위치를 보정합니다.
                ax.add_patch(patches.Ellipse(ellipse[0], ellipse[1][0], ellipse[1][1], angle=ellipse[2], fill=False,
                                             edgecolor='g'))
                #
                if 'BL' in name:
                    cv2.ellipse(img_draw, ellipse, (0, 0, 255), 1)
                else:
                    cv2.ellipse(img_draw, ellipse, (255, 0, 0), 1)

                center, (major_axis, minor_axis), angle = ellipse

                # 타원의 장축의 양 끝점 계산
                dx_major = major_axis / 2 * np.cos(np.deg2rad(angle))
                dy_major = major_axis / 2 * np.sin(np.deg2rad(angle))
                x_major = [center[0] - dx_major, center[0] + dx_major]
                y_major = [center[1] - dy_major, center[1] + dy_major]

                # 타원의 단축의 양 끝점 계산
                dx_minor = minor_axis / 2 * np.sin(np.deg2rad(angle))
                dy_minor = minor_axis / 2 * np.cos(np.deg2rad(angle))
                x_minor = [center[0] + dx_minor, center[0] - dx_minor]
                y_minor = [center[1] - dy_minor, center[1] + dy_minor]

                # 타원의 장축과 단축 그리기
                ax.plot(x_major, y_major, color='r', alpha=0.5, linewidth=1)
                ax.plot(x_minor, y_minor, color='b', alpha=0.5, linewidth=1)

    return img_draw, image


root = tk.Tk()
width_px = root.winfo_screenwidth()
height_px = root.winfo_screenheight()

# 모니터 해상도에 맞게 조절
mpl.rcParams['figure.dpi'] = 120  # DPI 설정
monitor_width_inches = width_px / mpl.rcParams['figure.dpi']  # 모니터 너비를 인치 단위로 변환
monitor_height_inches = height_px / mpl.rcParams['figure.dpi']  # 모니터 높이를 인치 단위로 변환

# Create a GridSpec object
gs = gridspec.GridSpec(2, 2)  # Change the number of rows and columns as needed

fig = plt.figure(figsize=(monitor_width_inches, monitor_height_inches), num='Compare Center')

# Create subplots using GridSpec
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])

dist0 = fig.add_subplot(gs[1, 0])
dist1 = fig.add_subplot(gs[1, 1])

# 이미지 파일 경로를 지정합니다.
blend_image_l = "./image_output/real_image/CAMERA_0_blender_test_image.png"
real_image_l = "../../../REAL_ROBOT/left_frame_1685066703.png"
# real_image_l = "./blended_image.png"
blend_image_r = "./image_output/real_image/CAMERA_1_blender_test_image.png"
real_image_r = "../../../REAL_ROBOT/right_frame_1685066703.png"



# 두 이미지를 블렌딩합니다.
B_img_1 = cv2.imread(blend_image_l)
R_img_1 = cv2.imread(real_image_l)
alpha_image_l = blend_images(ax0, "alpha_image_l.png", B_img_1, R_img_1, alpha=0.5)
cv2.imshow('B_img_1', B_img_1)
cv2.imshow('R_img_1', R_img_1)
B_img_r = cv2.imread(blend_image_r)
R_img_r = cv2.imread(real_image_r)
alpha_image_r = blend_images(ax1, "alpha_image_r.png", B_img_r, R_img_r, alpha=0.5)
cv2.imshow('B_img_r', B_img_r)
cv2.imshow('R_img_r', R_img_r)

# 결과 이미지를 표시합니다.
STACK_FRAME = np.hstack((alpha_image_l, alpha_image_r))
cv2.putText(STACK_FRAME, 'LEFT', (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(STACK_FRAME, 'RIGHT', (10 + CAP_PROP_FRAME_WIDTH, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

cv2.imshow('STACK Frame', STACK_FRAME)


key = cv2.waitKey(0)
cv2.destroyAllWindows()


# plt.show()
