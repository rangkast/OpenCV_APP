from typing import List, Tuple
import numpy as np
cimport numpy as cnp
import cv2

cdef void swap(cnp.int64_t[:] arr, int i, int j):
    cdef int temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp

cdef void permute(cnp.int64_t[:] data, int start, int end, list result):
    if start==end:
        result.append(np.copy(data))  # copy is necessary because we are swapping in-place
    else:
        for i in range(start, end+1):
            swap(data, start, i)
            permute(data, start+1, end, result)
            swap(data, i, start)  # backtrack

cdef list sliding_window(int data_len, int window_size):
    return [range(i, i + window_size) for i in range(data_len - window_size + 1)]

cpdef process_blobs(List[Tuple[float, float, Tuple[int, int, int, int]]] blobs, 
                    int window_size, 
                    int undistort, 
                    cnp.float64_t[:, :, :] camera_matrix,
                    cnp.float64_t[:, :] default_cameraK,
                    cnp.float64_t[:, :] default_dist_coeffs,
                    list SOLVE_PNP_FUNCTION,
                    int METHOD, 
                    cnp.float64_t[:, :] MODEL_DATA,
                    int CAM_ID):
    cdef int CNT = len(blobs)
    cdef int blob_idx, i, idx, BLOB_CNT = len(MODEL_DATA)
    cdef cnp.float64_t[:, :] points2D_D
    cdef cnp.float64_t[:, :] points2D_U
    cdef list candidates = []
    cdef list points3D_perm = []
    cdef cnp.float64_t[:, :] points3D
    cdef list INPUT_ARRAY = []
    cdef tuple output_tuple
    cdef str _, rvec, tvec
    cdef list perm_result = []

    blobs = sorted(blobs, key=lambda x:x[0]) 

    for blob_idx in sliding_window(CNT, window_size):
        candidates = [(blobs[i][0], blobs[i][1]) for i in blob_idx]
        points2D_D = np.array(candidates, dtype=np.float64)
        points2D_U = np.array(cv2.undistortPoints(points2D_D, camera_matrix[CAM_ID], default_dist_coeffs)).reshape(-1, 2)

        perm_result.clear()
        permute(np.arange(BLOB_CNT), 0, BLOB_CNT - 1, perm_result)

        for perm in perm_result:
            points3D_perm = [MODEL_DATA[idx] for idx in perm[:window_size]]
            points3D = np.array(points3D_perm, dtype=np.float64)

            INPUT_ARRAY = [
                CAM_ID,
                points3D,
                points2D_D if undistort == 0 else points2D_U,
                camera_matrix[CAM_ID] if undistort == 0 else default_cameraK,
                default_dist_coeffs
            ]
            _, rvec, tvec, _ = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
