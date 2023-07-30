from Advanced_Function import *


LED_NUMBER = [5, 4, 3, 2, 1, 0]

points2D = np.array([
    [565.23155435, 463.01475652],
    [595.81339823, 498.36512237],
    [633.40081453, 462.72414729],
    [668.31771687, 492.38420431],
    [695.78659295, 465.09464765],
    [723.63785228, 470.9473412 ]
])

points2D_U = np.array([
    [-0.1194495,  -0.03667688],
    [-0.07673795,  0.01278269],
    [-0.02425337, -0.03702632],
    [ 0.02457632,  0.00446988],
    [ 0.06300668, -0.03372991],
    [ 0.10191554, -0.02557575]
])

points3D = np.array([
    [0.02980453,  0.06146441,  0.01634117],
    [0.0417978,   0.035768,    0.01995002],
    [0.04281938,  0.02696581, -0.00181747],
    [0.03710598,  0.00931646,  0.00300296],
    [0.03201193,  0.00573596, -0.01205159],
    [0.02157939, -0.00317375, -0.01408719]
])

PnP_Solver_rvec = np.array([-1.27090419, 1.11495867, 1.90421112])
PnP_Solver_tvec = np.array([0.03012457, 0.00260204, 0.32740023])


# 정의된 점들과 내부 파라미터를 torch 텐서로 변환
points2D_U_torch = torch.from_numpy(points2D_U.astype(np.float32)).unsqueeze(0)
points3D_torch = torch.from_numpy(points3D.astype(np.float32)).unsqueeze(0)

# Default camera intrinsics 행렬 정의
intrinsics = torch.tensor([[[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]]], dtype=torch.float32)

# kornia 함수를 이용하여 카메라의 포즈 계산
world_to_cam = K.geometry.solve_pnp_dlt(points3D_torch, points2D_U_torch, intrinsics)

# The estimated world_to_cam matrices
print(world_to_cam)

# Extract rotation matrix from the world_to_cam tensor
rotation_matrix = world_to_cam[0, :3, :3].cpu().numpy()

# Convert rotation matrix to Rodrigues vector
rvec = cv2.Rodrigues(rotation_matrix)[0]

print(f"Rodrigues vector: {rvec}")

# Extract translation vector from the world_to_cam tensor
tvec = world_to_cam[0, :3, 3].cpu().numpy()

print(f"Translation vector: {tvec}")


# Here we assume `points3D`, `points2D`, `camera_k` and `dist_coeff` are already defined

# Calculate the reprojection error using the function you provided
rer_pnp_solver = reprojection_error(points3D, points2D, PnP_Solver_rvec, PnP_Solver_tvec, camera_matrix[0][0], camera_matrix[0][1])
print(f"PnP_Solver reprojection error: {rer_pnp_solver}")

# Convert PyTorch tensors to numpy arrays for the calculation
points3D_np = points3D_torch.cpu().numpy()
points2D_np = points3D_torch.cpu().numpy()

# Extract rvec and tvec from the result of Kornia PnP
rvec_kornia = cv2.Rodrigues(world_to_cam[0, :3, :3].cpu().numpy())[0]
tvec_kornia = world_to_cam[0, :3, 3].cpu().numpy()

# Calculate the reprojection error
rer_kornia = reprojection_error(points3D_np, points2D, rvec_kornia, tvec_kornia, camera_matrix[0][0], camera_matrix[0][1])
print(f"Kornia PnP reprojection error: {rer_kornia}")

