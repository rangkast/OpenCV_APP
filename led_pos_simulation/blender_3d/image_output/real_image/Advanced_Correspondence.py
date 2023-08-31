from Advanced_Function import *
frame_cnt = 15

# 2D 포인트 배열 지정
points2D = np.array([
    [576.70768249, 581.7040481 ],
    [504.72263609, 578.87014871],
    [650.10619452, 570.15952786],
    [463.10860692, 565.70369713],
    [571.69922548, 536.88958046],
    [500.61773961, 530.84227504],
    [654.59750173, 528.59149896],
    [700.32552378, 517.63362971],
    [457.67294478, 514.64425863]
])

# 정규화된 2D 포인트 배열 지정
points2D_U = np.array([
    [-0.10306651,  0.12863426],
    [-0.20281795,  0.12418817],
    [-0.0008731,   0.11283475],
    [-0.26024162,  0.10557942],
    [-0.11023781,  0.06642056],
    [-0.20887857,  0.05762731],
    [ 0.00538984,  0.0550232 ],
    [ 0.06927466,  0.03969615],
    [-0.26817362,  0.03483316]
])

points3D = np.array([
[-0.00538225, -0.03670253, 0.00442442],
[0.00978572, -0.04727607, 0.0040668],
[0.02993561, -0.05133603, 0.00403334],
[0.05145799, -0.0452342, 0.00322126],
[0.06952839, -0.02946931, 0.00289837],
[0.0774895, -0.01309678, 0.00305882],
[0.07833686, 0.00930992, 0.00274091],
[0.07173884, 0.02620113, 0.00300525],
[0.05323637, 0.0446104, 0.00333569],
[0.03440627, 0.05094375, 0.00376592],
[0.01247952, 0.04827114, 0.00409928],
[-0.00525621, 0.03652419, 0.00430552],
[-0.01725874, -0.02510102, 0.0170897],
[-0.00734298, -0.03632259, 0.01707089],
[0.02459884, -0.05074652, 0.01805065],
[0.04446367, -0.04730652, 0.01844551],
[0.06122225, -0.03579857, 0.01883048],
[0.07364986, -0.01468638, 0.01950387],
[0.07176939, 0.02063129, 0.01912847],
[0.0546692, 0.04178583, 0.01877653],
[0.03418668, 0.05023091, 0.01837584],
[0.01247775, 0.0485895, 0.01770159],
[-0.0073319, 0.03636108, 0.01712673],
[-0.01727314, 0.02493721, 0.01696496],
])



points3D_dir = np.array([
[-0.60688004, -0.75584205, -0.24576291],
[-0.33720495, -0.90003042, -0.27611241],
[ 0.11035048, -0.95576858, -0.27263381],
[ 0.51196981, -0.84107304, -0.17459397],
[ 0.83833736, -0.54169635, -0.06128241],
[ 0.96998041, -0.23536262, -0.06117552],
[ 0.98373745,  0.16887833, -0.06116157],
[ 0.87571599,  0.48070836, -0.04517721],
[ 0.54204116,  0.82528168, -0.15843461],
[ 0.19284405,  0.94275266, -0.27208197],
[-0.21963071,  0.92057199, -0.3229699 ],
[-0.60688004,  0.75584205, -0.24576291],
[-0.75344005, -0.64331363,  0.13592523],
[-0.64614776, -0.74783871,  0.152415  ],
[ 0.01024382, -0.93983665,  0.34147055],
[ 0.36842103, -0.8465559 ,  0.38419924],
[ 0.65268227, -0.62648053,  0.42606103],
[ 0.84769881, -0.25119497,  0.46723422],
[0.81854725, 0.35282319, 0.45331689],
[0.5400044 , 0.73368009, 0.41244245],
[0.18586076, 0.90365993, 0.38581669],
[-0.22076738,  0.92403974,  0.31210946],
[-0.64614776,  0.74783871,  0.152415  ],
[-0.80510213,  0.57732451,  0.13604035],    
])

# LED 번호 배열 지정
LED_NUMBER = np.array([10, 9, 11, 8, 21, 20, 22, 23, 19])

CAM_ID = 0

# KD-Tree 생성

points3D_tree = KDTree(points3D)
points2D_U_tree = KDTree(points2D_U)
points2D_tree = KDTree(points2D)
# 특정 LED를 기준으로 가까운 이웃 찾기 (예: 첫 번째 LED)

points3D_IND = []


print('points3D')
for i in range(len(points3D)):
    distances, indices = points3D_tree.query(points3D[i], k=4)  # 가장 가까운 4개의 이웃을 찾음
    print("distances", distances)
    print("indices", indices)
    points3D_IND.append(indices)

# print('points2D_U')
# for i in range(len(points2D_U)):
#     distances, indices = points2D_U_tree.query(points2D_U[i], k=4)  # 가장 가까운 4개의 이웃을 찾음
#     print("distances", distances)
#     print("indices", LED_NUMBER[indices])


print('points2D')
for i in range(len(points2D)):
    distances, indices = points2D_tree.query(points2D[i], k=4)  # 가장 가까운 4개의 이웃을 찾음
    print("distances", distances)
    print("indices", indices)

min_RER = float('inf')
min_points3d_ind = NOT_SET
min_points2d_ind = NOT_SET
min_rvec = NOT_SET
min_tvec = NOT_SET
for points3d_ind in points3D_IND:
    for i in range(len(points2D)):
        distances, points2d_ind = points2D_tree.query(points2D[i], k=4)  # 가장 가까운 4개의 이웃을 찾음

        POINTS3D = np.array(np.array(points3D[points3d_ind]), dtype=np.float64)
        POINTS2D = np.array(points2D[points2d_ind], dtype=np.float64)
        # print(f"POINTS3D {POINTS3D}")
        # print(f"POINTS2D {POINTS2D}")
        check_blob = POINTS2D[3]
        retval, rvec, tvec = cv2.solveP3P(POINTS3D[:3], POINTS2D[:3], camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][1], flags=cv2.SOLVEPNP_P3P)

        for ret_i in range(retval):
            RER, image_points = reprojection_error(POINTS3D,
                    POINTS2D,
                    rvec[ret_i],
                    tvec[ret_i],
                    camera_matrix[CAM_ID][0],
                    camera_matrix[CAM_ID][1])
            # if RER < 0.5:
            #     # print(image_points)
            #     # print('RER', RER)
            #     # print('points3D ind:', points3d_ind)
            #     # print('points2D ind:', LED_NUMBER[points2d_ind])
            #     # print('check_blob: ', check_blob)
            #     # print('candidates')
            if RER < min_RER:
                min_RER = RER
                min_points3d_ind = points3d_ind
                min_points2d_ind = points2d_ind
                min_rvec = rvec[ret_i]
                min_tvec = tvec[ret_i]

print(f"min_points3d_ind {min_points3d_ind}")
print(f"min_points2d_ind {min_points2d_ind}")
print(f"min_rvec {min_rvec}")
print(f"min_tvec {min_tvec}")


DP = [-1] * len(points2D)
MAP = [[-1,0]] * len(points3D)
for i, idx in enumerate(min_points2d_ind):
    DP[idx] = min_points3d_ind[i]
    MAP[min_points3d_ind[i]] = [idx, 0]

print(DP)
print(MAP)


# for led in DP:
#     if led != -1:
#         _, points3d_ind = points3D_tree.query(points3D[led], k=4)
#         print(points3d_ind[1:4])
#         for candidates in points3d_ind[1:4]:
#             if candidates not in DP:
#                 print(candidates)    

# show_calibrate_data(np.array(points3D), np.array(points3D_dir))


SEARCH_QUEUE = Queue()

while True:
    for led in DP:
        if led != -1:
            _, points3d_ind = points3D_tree.query(points3D[led], k=4)
            print(points3d_ind[1:4])            
            for candidates in points3d_ind[1:4]:
                if MAP[candidates][0] != -1:
                    continue
                print(f"candidates {candidates} MAP {MAP[candidates]}")

                if MAP[candidates][1] == 0:
                    SEARCH_QUEUE.put(candidates)
                    print(f"put {candidates}")
                    # 여기 왜 이런지 모르겠음
                    MAP[candidates] = [-1, 1]
        MAP[led][1] = 2


    item = SEARCH_QUEUE.get()
    print(f"get {item}")

    # Add Searching
    points3D_candidates = np.array(points3D[item], dtype=np.float64)
    print(f"points3D_candidates {points3D_candidates}")
    min_RER = float('inf')
    min_i = -1
    for i in range(len(points2D)):
        if DP[i] != -1:
            continue

        points2D_candidates = np.array(points2D[i], dtype=np.float64)
        print(f"points2D_candidates {points2D_candidates}")
        RER, image_points = reprojection_error(points3D_candidates,
                                               points2D_candidates,
                                               min_rvec,
                                               min_tvec,
                                               camera_matrix[CAM_ID][0],
                                               camera_matrix[CAM_ID][1])
        print(f"i {i} RER {RER}")
        if RER < min_RER:
            min_RER = RER
            min_i = i

    DP[min_i] = item

    print(f"DP {DP}")
    print(f"MAP {MAP}")

    if SEARCH_QUEUE.empty():        
        break





# def parametric_search(arr, target):
#     left = 0
#     right = len(arr) - 1
#     result = -1

#     while left <= right:
#         mid = (left + right) // 2

#         if arr[mid]>= target:
#             result = arr[mid]
#             right = mid - 1
#         else:
#             left = mid + 1

#     return result

# # 정렬된 배열
# arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

# # 파라미터 검색 수행
# target = 10
# result = parametric_search(arr, target)

# # 결과 출력
# if result != -1:
#     print(f"가장 작은 값: {result}")
# else:
#     print("조건을 만족하는 값이 없습니다.")
