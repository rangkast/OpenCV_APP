import cv2
import numpy as np

# 이미지 로딩
image = cv2.imread('./blender_test_image.png')

# 흰색 픽셀 찾기: 흰색을 표현하는 RGB 값은 (255, 255, 255)입니다.
# cv2.inRange 함수를 사용하여 흰색 픽셀만 이진 이미지로 추출합니다.
white_pixels = cv2.inRange(image, (255, 255, 255), (255, 255, 255))

# 윤곽선 찾기
contours, _ = cv2.findContours(white_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 각 윤곽선에 대해 중심점 계산
for contour in contours:
    # 윤곽선의 모멘트를 계산합니다.
    M = cv2.moments(contour)

    # 0으로 나누는 경우를 방지하기 위해 m00이 0인지 확인합니다.
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # 중심점을 출력합니다.
    print("Center: ({}, {})".format(cX, cY))
