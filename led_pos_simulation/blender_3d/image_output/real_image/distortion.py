import numpy as np
import cv2

camera_matrix = [
    [np.array([[712.623, 0.0, 653.448],
               [0.0, 712.623, 475.572],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)],
    [np.array([[716.896, 0.0, 668.902],
               [0.0, 716.896, 460.618],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.07542], [-0.026874], [0.006662], [-0.000775]], dtype=np.float64)]
]
default_dist_coeffs = np.zeros((4, 1))
default_cameraK = np.eye(3).astype(np.float64)
blend_image_l = "./CAMERA_0_blender_test_image.png"
real_image_l = "./left_frame.png"
blend_image_r = "./CAMERA_1_blender_test_image.png"
real_image_r = "./right_frame.png"


# 렌더링 이미지 로드
img = cv2.imread(blend_image_l)

# 카메라 매트릭스와 왜곡 계수 설정
camera_matrix = camera_matrix[0][0]
dist_coeffs = camera_matrix[0][1]


# 이미지 크기를 가져옵니다.
h, w = img.shape[:2]

# 새 카메라 행렬을 계산합니다.
newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0, (w, h))

# 왜곡된 이미지를 계산합니다.
distorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramatrix)

# 왜곡된 이미지를 출력합니다.
cv2.imwrite('distorted_image.png', distorted_img)
