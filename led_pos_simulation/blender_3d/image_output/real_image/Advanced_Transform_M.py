import numpy as np
import cv2


'''
1. 실제 로봇에서 단순 angle 변환으로는  컨트롤러의 축변환을 적용할 수 없음, 
    즉, 컨트롤러의 축을 변환 시키면 실제 카메라는 rvec. tvec이 모두 변환되어야 함
2. 로봇시스템에서 컨트롤러 축변환을 하면 월드좌표계에서 컨트롤러와 카메라 모두 축이 일치되어 있지 않아
    translation matrix 가 적용되지 않는다.
3. 만약 초기 R|T를 구한다고 하더라도 서로의 z축이 평행하지 않는다면 회전시 축이 틀어질 것
4. 매번 컨트롤러가 놓일 때마다 projectPoint를 하면 값이 조금씩 틀어질 텐데 이 오차처리가 애매함
'''

# Pose A
rvec_B = np.array([-0.89945821, 0.65749896, 1.13889868])
tvec_B = np.array([0.02544914, 0.03041541, 0.38084769])

# Pose B
rvec_A = np.array([-1.33942214, 0.97914561, 0.97919369])
tvec_A = np.array([0.0254568, 0.01676413, 0.31461424])

# Convert rotation vectors to rotation matrices
R_A, _ = cv2.Rodrigues(rvec_A)
R_B, _ = cv2.Rodrigues(rvec_B)

# Compose the transformation matrices
T_A = np.hstack((R_A, tvec_A.reshape(-1,1)))
T_A = np.vstack((T_A, [0,0,0,1]))

T_B = np.hstack((R_B, tvec_B.reshape(-1,1)))
T_B = np.vstack((T_B, [0,0,0,1]))

# Compute the transformation from A to B
T_AtoB = np.linalg.inv(T_A) @ T_B

# Extract the rotation part
R_AtoB = T_AtoB[:3,:3]

# Extract the translation part
t_AtoB = T_AtoB[:3,3]

# Convert rotation matrix to euler angles
def rotationMatrixToEulerAngles(R) :
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


euler_AtoB = rotationMatrixToEulerAngles(R_AtoB)

# Now euler_AtoB contains the rotation from A to B in Euler angles (pitch, yaw, roll)
# and t_AtoB contains the translation from A to B

print("Euler Angles: ", euler_AtoB)
print("Translation: ", t_AtoB)

# New A pose
rvec_A = np.array([-1.35771571, 0.941488, 0.94201421])
tvec_A = np.array([0.02702102, 0.01677251, 0.31580645])

# Convert rotation vector of new A pose to rotation matrix
R_A, _ = cv2.Rodrigues(rvec_A)

# Compose the transformation matrix for the new A pose
T_A = np.hstack((R_A, tvec_A.reshape(-1,1)))
T_A = np.vstack((T_A, [0,0,0,1]))

# Get new B pose by applying transformation matrix from A to B
T_new_B = T_A @ T_AtoB

# Extract the rotation part
R_new_B = T_new_B[:3, :3]

# Extract the translation part
t_new_B = T_new_B[:3, 3]

# Convert rotation matrix to rotation vector
rvec_new_B, _ = cv2.Rodrigues(R_new_B)

print("New B pose:")
print("Rvec: ", rvec_new_B)
print("Tvec: ", t_new_B)



# Original B pose
rvec_B = np.array([-1.33942214, 0.97914561, 0.97919369])
tvec_B = np.array([0.0254568, 0.01676413, 0.31461424])

# Convert rotation vector to rotation matrix
R_B, _ = cv2.Rodrigues(rvec_B)

# Create a rotation matrix for a 30 degree rotation about x-axis
rotation_angle = np.deg2rad(30)  # convert degree to radian
R_x = np.array([[1, 0, 0],
                [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
                [0, np.sin(rotation_angle), np.cos(rotation_angle)]])

# Apply the rotation to the existing rotation matrix
R_B_tilted = R_x @ R_B

# Convert the updated rotation matrix back to a rotation vector
rvec_B_tilted, _ = cv2.Rodrigues(R_B_tilted)

print("Updated B pose:")
print("Rvec: ", rvec_B_tilted)
print("Tvec: ", tvec_B)
