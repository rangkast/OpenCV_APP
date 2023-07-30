# distutils: language=c++

from typing import List, Tuple
import numpy as np
# from itertools import combinations
cimport numpy as cnp
# Get the directory of the current script
import poselib

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

cdef void combinations_util(cnp.int64_t[:] arr, cnp.int64_t[:] data, int start, int end, int index, int r, list res):
    cdef int i
    # Current combination is ready to be printed, print it
    if index == r:
        res.append(np.copy(data[0:r]))
        return
    # replace index with all possible elements. The condition
    # "end-i+1 >= r-index" makes sure that including one element
    # at index will make a combination with remaining elements
    # at remaining positions
    i = start 
    while(i <= end and end - i + 1 >= r - index):
        data[index] = arr[i]
        combinations_util(arr, data, i+1, end, index+1, r, res)
        i += 1

cpdef list combinations(cnp.int64_t[:] arr, int r):
    #print("combinations: input: ", arr, " r: ", r)
    cdef:
        cnp.int64_t[:] data = np.empty(r, dtype=np.int64)
        list res = []
    combinations_util(arr, data, 0, len(arr)-1, 0, r, res)
    #print("combinations: result: ", res)  # print the result
    return res


cdef list sliding_window(int data_len, int window_size):
    return [range(i, i + window_size) for i in range(data_len - window_size + 1)]

cpdef list circular_sliding_window(list data, int window_size):
    data = data + data[:window_size-1]
    return [data[i:i + window_size] for i in range(len(data) - window_size + 1)]

cpdef list cython_func(cnp.ndarray[cnp.float64_t, ndim=2] MODEL_DATA, cnp.ndarray[cnp.float64_t, ndim=2] points2D_U, int BLOB_CNT, int SEARCHING_WINDOW_SIZE, int BLOBS_LENGTH):
    cdef:
        cnp.ndarray[cnp.float64_t, ndim=2] points2d_u
        cnp.ndarray[cnp.float64_t, ndim=1] MIN_GROUP_ID = np.empty((0,), dtype=np.float64)
        cnp.ndarray[cnp.float64_t, ndim=2] points3D = np.empty((0,0), dtype=np.float64), points3D_grp
    cdef:
        dict camera = {'model': 'SIMPLE_PINHOLE', 'width': 1280, 'height': 960, 'params': [715.159, 650.741, 489.184]}
        float MIN_SOCRE = float('inf')
        object MIN_POSE = None
        dict MIN_INFO = {}
        object pose
        dict info
        list grps
    #print("BLOB_CNT: ", BLOB_CNT, "points2D_U: ", points2D_U, "BLOBS_LENGTH: ", BLOBS_LENGTH)
    for grps in circular_sliding_window(list(range(BLOB_CNT)), SEARCHING_WINDOW_SIZE):
        #print("grps: ", grps)
        for points3D_grp_comb in combinations(np.array(grps, dtype=np.int64), BLOBS_LENGTH):
            #print("points3D_grp_comb: ", points3D_grp_comb) # combination 출력
            points3D_grp = MODEL_DATA[list(points3D_grp_comb), :]
            #print("points3D_grp: ", points3D_grp)
            for i in range(0, points2D_U.shape[0] - BLOBS_LENGTH + 1):
                points2d_u = points2D_U[i:i+BLOBS_LENGTH]                                             
                pose, info = poselib.estimate_absolute_pose(points2d_u, points3D_grp, camera, {'max_reproj_error': 10.0}, {})
                if info['model_score'] < MIN_SOCRE:
                    MIN_SOCRE = info['model_score']
                    MIN_POSE = pose
                    MIN_INFO = info
                    MIN_GROUP_ID = np.array(points3D_grp_comb, dtype=np.float64)
                    points3D = points3D_grp
                    '''
                    print("New minimum score found: ", MIN_SOCRE)
                    print("Associated pose: ", MIN_POSE)
                    print("Associated info: ", MIN_INFO)
                    print("Associated group ID: ", MIN_GROUP_ID)
                    print("Associated 3D points: ", points3D)
                    '''
    return [MIN_POSE, MIN_INFO, MIN_GROUP_ID, points3D]
