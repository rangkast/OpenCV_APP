import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

points3D = np.array([[-5.0, 1.0, 1.0], [-5.0, 1.0, -1.0], [-5.0, -1.0, 1.0], [-5.0, -1.0, -1.0]])
noise = np.random.normal(0, 0.1, points3D.shape)  # mean = 0, std_dev = 0.1
distorted_points3D = points3D + noise
print('distorted_points3D\n', distorted_points3D)
default_cameraK = np.eye(3).astype(np.float64)
distCoeffs = np.zeros(4)  # 왜곡 계수는 0으로 가정
num_cameras = 2
RTs = [cv2.Rodrigues(np.random.randn(3))[0] for _ in range(num_cameras)]  # 임의의 회전
Ts = [np.random.randn(3) for _ in range(num_cameras)]  # 임의의 이동
distorted_points2D = []
for R, T in zip(RTs, Ts):
    projected_points, _ = cv2.projectPoints(distorted_points3D, R, T, default_cameraK, distCoeffs)
    distorted_points2D.append(projected_points.reshape(-1, 2))
estimated_RTs = []
for points2D in distorted_points2D:
    ret, rvec, tvec = cv2.solvePnP(points3D, points2D, default_cameraK, distCoeffs)
    estimated_RTs.append((rvec.ravel(), tvec.ravel()))
print('estimated_RTs')
for estimated_RT in estimated_RTs:
    print(estimated_RT)


def fun(params, n_cameras, n_points, camera_indices, points_indices, points_2d):
    points_3d = params[:n_points * 3].reshape((n_points, 3))
    camera_params = params[n_points * 3:].reshape((n_cameras, 6))
    points_proj = np.zeros(points_2d.shape)
    # print(points_3d)
    for i in range(points_2d.shape[0]):
        camera_index = camera_indices[i]
        point_index = points_indices[i]
        R, _ = cv2.Rodrigues(camera_params[camera_index, :3])
        T = camera_params[camera_index, 3:]
        points_proj[i], _ = cv2.projectPoints(points_3d[point_index].reshape(1, 1, 3), R, T, default_cameraK,
                                              distCoeffs)
    return (points_proj - points_2d).ravel()


n_cameras = len(estimated_RTs)
n_points = len(points3D)
camera_indices = np.repeat(np.arange(n_cameras), n_points)
points_indices = np.tile(np.arange(n_points), n_cameras)
print('n_cameras', n_cameras, 'n_points', n_points)
print('camera_indices', camera_indices)
print('points_indices', points_indices)
points_2d = np.concatenate(distorted_points2D)
print('points_2d', points_2d)

# 수정한 코드
estimated_RTs_flattened = [np.hstack(rt) for rt in estimated_RTs]

print(points3D.ravel())
print(estimated_RTs_flattened)
x0 = np.hstack((points3D.ravel(), np.hstack(estimated_RTs_flattened)))


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3  # Note: Here camera parameters are 6 (3 for rotation and 3 for translation)
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(6):  # Changed here from 9 to 6
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A
A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, points_indices)
res = least_squares(fun, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    jac_sparsity=A, args=(n_cameras, n_points, camera_indices, points_indices, points_2d))
# res = least_squares(fun, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
#                     args=(n_cameras, n_points, camera_indices, points_indices, points_2d))
estimated_points3D = res.x[:n_points * 3].reshape((n_points, 3))
print("오염된 3D 좌표:")
print(estimated_points3D)
ret, out, inliers = cv2.estimateAffine3D(points3D, estimated_points3D)
R = out[:, :3]
T = out[:, 3]
print("변환 행렬:")
print("Rotation matrix:\n", R)
print("Translation vector:\n", T)
restored_points3D = cv2.transform(points3D.reshape(-1, 1, 3), out).reshape(-1, 3)
print("복원된 3D 좌표:")
print(restored_points3D)
