import cv2
import matplotlib.pyplot as plt

# 이미지 로드
image = cv2.imread('right_frame.png')

# Gaussian 피라미드 생성
smaller = cv2.pyrDown(image)
larger = cv2.pyrUp(smaller)

# 이미지 출력
plt.figure(figsize=(12, 3))
plt.subplot(131)
plt.title('Original')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(132)
plt.title('Smaller')
plt.imshow(cv2.cvtColor(smaller, cv2.COLOR_BGR2RGB))
plt.subplot(133)
plt.title('Larger')
plt.imshow(cv2.cvtColor(larger, cv2.COLOR_BGR2RGB))
plt.show()
