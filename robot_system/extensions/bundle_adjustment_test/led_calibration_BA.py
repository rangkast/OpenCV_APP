from definition_BA import *

LED_CNT = 5
N_CAM_PARAMS = 6


if __name__ == '__main__':
    print(TAG, 'START')
    ransac_dump_file_5points = 'dump_20230208_140916.pickle'
    ransac_dump_file_6points = 'dump_20230209_084229.pickle'
    ransac_dump_file_5points_line = 'dump_20230209_154033.pickle'
    ransac_dump_file_5points_rot = 'dump_20230209_155718.pickle'
    file = ''.join(['../../jsons/dump/', ransac_dump_file_5points_line])
    dump_data = pickle_data(READ, file, None)

    plt.style.use('default')
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111, projection='3d', title='3D_point')

    points_origin, drt_remake, srt_remake, rigid_remake = remake_3d_points(dump_data)

    print('origin\n', points_origin)
    test_draw(ax, [points_origin, 'points_origin', 'blue'])
    # test_draw(ax, [drt_remake, 'drt_remake', 'black'])
    # test_draw(ax, [srt_remake, 'srt_remake', 'black'])
    # test_draw(ax, [rigid_remake, 'rigid_remake', 'green'])

    ax.scatter(0, 0, 0, marker='o', color='purple', s=20)
    ax.set_xlim([-0.7, 0.7])
    ax.set_xlabel('X')
    ax.set_ylim([-0.7, 0.7])
    ax.set_ylabel('Y')
    ax.set_zlim([-0.7, 0.7])
    ax.set_zlabel('Z')
    scale = 1.5
    f = zoom_factory(ax, base_scale=scale)

    group_cnt = len(dump_data[GROUP_DATA_INFO])
    n_observations = group_cnt * LED_CNT
    points_2d_l = []
    points_2d_r = []
    drt_array_l = []
    drt_array_r = []

    for cidx, mdata in enumerate(dump_data[MEASUREMENT_INFO]):
        curr_blob = mdata.get_blob().get_curr_blobs()
        undistort_2d = mdata.get_blob().get_undistort_2d()
        mdata_rt = mdata.get_r_t().get_curr_r_t()
        if len(curr_blob) > 0:
            if cidx % 2 == 0:
                drt_array_l.append(mdata_rt)
                for i, blob_2d in enumerate(curr_blob):
                    points_2d_l.append(undistort_2d[i].ravel())
            else:
                drt_array_r.append(mdata_rt)
                for i, blob_2d in enumerate(curr_blob):
                    points_2d_r.append(undistort_2d[i].ravel())


    print('group_cnt ', group_cnt)
    print('n_observations ', n_observations)

    points_2d_l = np.array(points_2d_l)
    print_array_info(NAME(points_2d_l, 'points_2d_l'))

    '''
    TEST 1
    cam id 0 1 0 2 0 3 0 4 .....
    left 2d points    
    '''

    # make 3d points
    points_3d = []
    points_i = [[ii for ii in range(LED_CNT)]]
    camera_i = [[0 for ii in range(LED_CNT)]]
    D3D_DLT = []
    for i in range(1, len(drt_array_l)):
        Rod_0, _ = cv2.Rodrigues(drt_array_l[0].rvecs)
        Rod_n, _ = cv2.Rodrigues(drt_array_l[i].rvecs)

        P_0 = np.hstack((Rod_0, drt_array_l[0].tvecs))
        P_n = np.hstack((Rod_n, drt_array_l[i].tvecs))
        points_i.append([(i - 1)*LED_CNT + ii for ii in range(LED_CNT)])
        camera_i.append([i for ii in range(LED_CNT)])
        points_3d.append(K.geometry.triangulate_points(P_0, P_n, AtoT(points_2d_l[0:LED_CNT]),
                                                       AtoT(points_2d_l[i * LED_CNT:i * LED_CNT + LED_CNT])))

        # Test fundamental M
        F = get_fundamental_matrix(points_2d_l[0:LED_CNT], points_2d_l[i * LED_CNT:i * LED_CNT + LED_CNT])
        E = default_cameraK.T.dot(F).dot(default_cameraK)
        _, R, T, M = cv2.recoverPose(E, points_2d_l[0:LED_CNT], points_2d_l[i * LED_CNT:i * LED_CNT + LED_CNT])

        # RT matrix for C1 is identity.
        RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
        P_1 = default_cameraK @ RT1
        RT2 = np.concatenate([R, T], axis=-1)
        P_2 = default_cameraK @ RT2

        for ii in range(LED_CNT):
            triangulation = DLT(P_1, P_2, np.array(points_2d_l[0:LED_CNT][ii]), np.array(points_2d_l[i * LED_CNT:i * LED_CNT + LED_CNT][ii]))
            D3D_DLT.append(triangulation)
    test_draw(ax, [D3D_DLT, 'D3D_DLT', 'purple'])

    points_indices = TtoA(points_i)
    camera_indices = TtoA(camera_i)
    print_array_info(NAME(camera_indices, 'camera_indices'))
    # print_array_info(NAME(np.array(drt_array_l), 'drt_array_l'))
    print_array_info(NAME(points_indices, 'points_indices'))
    points_3d = TtoA(points_3d)
    print_array_info(NAME(points_3d, 'points_3d'))
    # test_draw(ax, [points_3d, 'points_3d', 'purple'])

    n_cameras = len(drt_array_l)
    n_points = len(points_3d)
    # Test Start
    # make camera param array
    camera_params = np.empty((n_cameras, N_CAM_PARAMS))
    for i, mdata in enumerate(drt_array_l):
        camera_params[i] = [float(mdata.rvecs[0][0]),
                            float(mdata.rvecs[1][0]),
                            float(mdata.rvecs[2][0]),
                            float(mdata.tvecs[0][0]),
                            float(mdata.tvecs[1][0]),
                            float(mdata.tvecs[2][0])]
    print_array_info(NAME(camera_params, 'camera_params'))

    # test
    # x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    x0 = np.hstack((camera_params.ravel(), np.array(D3D_DLT).ravel()))

    f0 = fun(x0, n_cameras, n_points, camera_indices, points_indices, points_2d_l)
    print('f0\n', f0)
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, points_indices)
    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-8, method='trf',
                        args=(n_cameras, n_points, camera_indices, points_indices, points_2d_l))
    t1 = time.time()
    n_cam_params = res.x[:len(camera_params.ravel())].reshape((n_cameras, N_CAM_PARAMS))
    n_points_3d = res.x[len(camera_params.ravel()):].reshape((n_points, 3))
    test_draw(ax, [n_points_3d, 'n_points_3d', 'green'])

    origin_pts = []
    before_pts = []
    for i in range(LED_CNT):
        origin_pts.append(points_3d[i])
        before_pts.append(n_points_3d[i])

    origin_pts = np.array(origin_pts)
    before_pts = np.array(before_pts)

    # make 3xN matrix
    A = np.array([[0 for j in range(LED_CNT)] for i in range(3)], dtype=float)
    B = np.array([[0 for j in range(LED_CNT)] for i in range(3)], dtype=float)
    for r in range(LED_CNT):
        for c in range(3):
            B[c][r] = origin_pts[r][c]
            A[c][r] = before_pts[r][c]

    ret_R, ret_t = rigid_transform_3D(A, B)
    C = (ret_R @ A) + ret_t
    n_rigid_3d = []
    print('C\n', C)
    for r in range(LED_CNT):
        n_rigid_3d.append([float(C[0][r]), float(C[1][r]), float(C[2][r])])

    test_draw(ax, [n_rigid_3d, 'n_rigid_3d', 'red'])

    plt.show()
