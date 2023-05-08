import cv2
import numpy as np

def blend_images(image1, image2, alpha=0.5):
    # 이미지를 읽고, 동일한 크기로 조정합니다.
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 이미지를 alpha 비율로 블렌딩합니다.
    blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

    return blended

# 이미지 파일 경로를 지정합니다.
image1 = "./render_output/CAMERA_0_x163_y-42_z-52_20230508_163301.png"
image2 = "./render_output/left_frame_1683529007.png"



# 두 이미지를 블렌딩합니다.
blended_image = blend_images(image1, image2, alpha=0.5)
# # 필요한 경우 결과 이미지를 저장합니다.
cv2.imwrite("blended_image.png", blended_image)
# 결과 이미지를 표시합니다.
cv2.imshow("Blended Image", blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#

