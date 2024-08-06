
import numpy as np
from sklearn.decomposition import PCA
from collections import OrderedDict
import time
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import cv2

from Advanced_Function import *

def BA_RT(**kwargs):
    CAM_ID = 0
    undistort = 1
    info_name = kwargs.get('info_name')    
    save_to = kwargs.get('save_to')
    target = kwargs.get('target')
    print('BA_RT START')
    CAMERA_INFO = pickle_data(READ, info_name, None)[info_name.split('.')[0]]   
    camera_indices = []
    point_indices = []
    estimated_RTs = []
    POINTS_2D = []
    POINTS_3D = []
    n_points = 0
    cam_id = 0

    if len(CAMERA_INFO) <= 0:
        return ERROR
    LED_INDICES = []
    for frame_cnt, cam_info in CAMERA_INFO.items():
        # print(cam_info['LED_NUMBER'])
        # if len(cam_info['LED_NUMBER']) <= 0:
        #     continue        
        # if cam_info[target]['status'] == NOT_SET:
        #     continue
        points3D = cam_info['points3D']
        # 여기 다시 확인 해야 함
        rvec = cam_info[target]['rt']['rvec']
        tvec = cam_info[target]['rt']['tvec']
        points2D_D = cam_info['points2D']['greysum']
        points2D_U = cam_info['points2D_U']['greysum']
        if len(cam_info['LED_NUMBER']) <= 3:
            print('led count is <= 3 ', cam_info['LED_NUMBER'])
            print('frame_cnt ', frame_cnt)
            print('rvec ', rvec)
            print('tvec ', tvec)   
            
        # Add camera parameters (rvec and tvec)
        estimated_RTs.append((rvec.ravel(), tvec.ravel()))

        # Adding 2D points
        POINTS_2D.extend(points2D_D if undistort == 0 else points2D_U)
        
        # Adding 3D points
        POINTS_3D.extend(points3D)

        LED_INDICES.extend(cam_info['LED_NUMBER'])
        
        # Adding indices
        camera_indices.extend([cam_id]*len(points3D))
        point_indices.extend(list(range(n_points, n_points+len(points3D))))

        n_points += len(points3D)
        cam_id += 1

    def fun(params, n_cameras, camera_indices, point_indices, points_2d, points_3d, camera_matrix):
        camera_params = params.reshape((n_cameras, 6))
        points_proj = []

        for i, POINT_3D in enumerate(points_3d[point_indices]):
            camera_index = camera_indices[i]
            rvec = camera_params[camera_index, :3]
            tvec = camera_params[camera_index, 3:]
            POINT_2D_PROJ, _ = cv2.projectPoints(POINT_3D,
                                                 np.array(rvec),
                                                 np.array(tvec),
                                                 camera_matrix[CAM_ID][0] if undistort == 0 else default_cameraK,
                                                 camera_matrix[CAM_ID][1] if undistort == 0 else default_dist_coeffs)
            points_proj.append(POINT_2D_PROJ[0][0])

        points_proj = np.array(points_proj)
        return (points_proj - points_2d).ravel()

    def bundle_adjustment_sparsity_RT(n_cameras, camera_indices):
        m = camera_indices.size * 2
        n = n_cameras * 6
        A = lil_matrix((m, n), dtype=int)
        i = np.arange(camera_indices.size)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1
        return A
    
    def bundle_adjustment_sparsity_Point(n_points, point_indices):
        m = point_indices.size * 2
        n = n_points * 3
        A = lil_matrix((m, n), dtype=int)
        i = np.arange(point_indices.size)
        for s in range(3):
            A[2 * i, point_indices * 3 + s] = 1
            A[2 * i + 1, point_indices * 3 + s] = 1
        return A
    
    def bundle_adjustment_sparsity_Point_RT(n_cameras, n_points, camera_indices, point_indices):
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


    # Convert the lists to NumPy arrays
    n_cameras = len(estimated_RTs)
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    camera_params = np.array(estimated_RTs).reshape(-1, 6)
    POINTS_2D = np.array(POINTS_2D).reshape(-1, 2)
    POINTS_3D = np.array(POINTS_3D).reshape(-1, 3)

    x0 = camera_params.ravel()
    A = bundle_adjustment_sparsity_RT(n_cameras, camera_indices)
    # A = bundle_adjustment_sparsity_Point_RT(n_cameras, n_points, camera_indices, point_indices)
    
    print('\n')
    print('#################### BA  ####################')
    print('n_points', n_points)
    print(f"A: {A.shape[0]}")

    print('n_cameras', n_cameras)
    print('len(point_indices)', len(point_indices))
    print('len(camera_params)', len(camera_params))
    print('len(camera_indices)', len(camera_indices))
    print(f"camera_params\n{camera_params[:50]}")
    print(f"point_indices\n{point_indices[:50]}")
    print(f"camera_indices\n{camera_indices[:50]}")

    # Visualize the sparse matrix
    plt.figure(figsize=(20, 20))
    plt.spy(A, markersize=1, aspect='auto')
    plt.show()

    start_time = time.time() 
    # Use Sparsity pattern
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-6, method='trf',
                        args=(n_cameras, camera_indices, point_indices, POINTS_2D, POINTS_3D, camera_matrix))

    # res = least_squares(fun, x0, verbose=2, x_scale='jac', ftol=1e-6, method='trf',
    #                     args=(n_cameras, camera_indices, point_indices, POINTS_2D, POINTS_3D, camera_matrix))

    end_time = time.time()  # Store the current time again
    elapsed_time = end_time - start_time  # Calculate the difference
    print(f"The function took {elapsed_time} seconds to complete.")         

    # You are only optimizing camera parameters, so the result only contains camera parameters data
    n_cameras_params = res.x.reshape((n_cameras, 6))
    # print("Optimized camera parameters: ", n_cameras_params, ' ', len(n_cameras_params))
    # n_points_3d = res.x.reshape((n_points, 3))

    file = save_to
    data = OrderedDict()
    data['BA_RT'] = n_cameras_params
    data['camera_indices'] = camera_indices
    # data['BA_3D'] = n_points_3d
    data['LED_INDICES'] = LED_INDICES
    # ret = pickle_data(WRITE, file, data)
    # if ret != ERROR:
    #     print('data saved')
