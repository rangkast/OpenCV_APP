import cv2
import numpy as np

# 이미지 로딩
image = cv2.imread('./blender_test_image.png')
ret, img_filtered = cv2.threshold(image, 100, 255, cv2.THRESH_TOZERO)
IMG_GRAY = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
# 윤곽선 찾기
contours, _ = cv2.findContours(IMG_GRAY, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 각 윤곽선에 대해 중심점 계산
for contour in contours:
    # 윤곽선의 모멘트를 계산합니다.
    M = cv2.moments(contour)

    # 0으로 나누는 경우를 방지하기 위해 m00이 0인지 확인합니다.
    if M["m00"] != 0:
        cX = float(M["m10"] / M["m00"])
        cY = float(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # 중심점을 출력합니다.
    print("Center: ({}, {})".format(cX, cY))

    # 중심점에 작은 빨간점 그리기
    cv2.circle(image, (int(cX), int(cY)), 1, (0, 0, 255), -1)  # 빨간색은 BGR에서 (0, 0, 255)입니다.

# 결과 이미지 보기
cv2.imshow("Image with center points", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
