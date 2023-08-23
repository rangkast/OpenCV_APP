from Advanced_Function import *
from numba import jit, float64, int32

ANGLE = 3

# RIGHT
CALIBRATION_DATA = np.array([
[-0.00538687, -0.03667682, 0.00445772],
[0.0098022, -0.0473034, 0.00409117],
[0.02997389, -0.05136959, 0.00401406],
[0.05154983, -0.04530395, 0.00328682],
[0.06954976, -0.0294302, 0.00290234],
[0.07745341, -0.01302696, 0.00305278],
[0.07828176, 0.00932429, 0.00279012],
[0.07171878, 0.02618578, 0.00302571],
[0.0532368, 0.04464043, 0.00334939],
[0.03441114, 0.05096652, 0.00374139],
[0.01247131, 0.0483145, 0.00409996],
[-0.00525508, 0.03654212, 0.00433411],
[-0.01722613, -0.02505649, 0.01704023],
[-0.00733975, -0.03627484, 0.01708449],
[0.02456592, -0.05078352, 0.01800659],
[0.0444748, -0.04733696, 0.01847506],
[0.06123124, -0.03581618, 0.0188381],
[0.07360234, -0.01468775, 0.01939855],
[0.07167104, 0.0205667, 0.01912662],
[0.05465507, 0.0417245, 0.01875003],
[0.03420542, 0.05024073, 0.01834996],
[0.01250846, 0.04858537, 0.01770699],
[-0.00730424, 0.03634569, 0.01714348],
[-0.01726364, 0.02494987, 0.01695512],
])

# LEFT
# CALIBRATION_DATA = np.array([
# [-0.00529076, 0.03664364, 0.00450423],
# [0.00988355, 0.04736619, 0.00423116],
# [0.0299106, 0.05138084, 0.00387957],
# [0.05172451, 0.04555571, 0.00345301],
# [0.06962105, 0.02943317, 0.00311068],
# [0.07749717, 0.01306848, 0.00261047],
# [0.0784388, -0.00957054, 0.00309278],
# [0.07184185, -0.02634136, 0.00306276],
# [0.0531229, -0.04452448, 0.00327469],
# [0.03446714, -0.05094538, 0.00344036],
# [0.01257801, -0.04854011, 0.00403444],
# [-0.00531085, -0.03665892, 0.00466532],
# [-0.01725951, 0.02493791, 0.01687879],
# [-0.00748959, 0.03618384, 0.01731351],
# [0.02447184, 0.05078297, 0.01767104],
# [0.04444158, 0.0471655, 0.0186316],
# [0.0613711, 0.03582667, 0.01875855],
# [0.07378114, 0.01479769, 0.01929112],
# [0.07142573, -0.02043988, 0.01929378],
# [0.05418478, -0.04120638, 0.0187185],
# [0.03417434, -0.05017958, 0.01818904],
# [0.01240688, -0.04883058, 0.01782703],
# [-0.0072397, -0.03638084, 0.01725842],
# [-0.01716512, -0.02484437, 0.01682995],
# ])

# ToDo

TARGET_DEVICE = 'ARCTURAS'
ORIGIN_DATA, DIRECTION = init_coord_json(f"{script_dir}/jsons/specs/arcturas_right.json")
MODEL_DATA, DIRECTION = init_coord_json(f"{script_dir}/jsons/specs/arcturus_#3_right+.json")
# ORIGIN_DATA, DIRECTION = init_coord_json(f"{script_dir}/jsons/specs/arcturas_left.json")
# MODEL_DATA, DIRECTION = init_coord_json(f"{script_dir}/jsons/specs/arcturus_#3_left.json")
CAMERA_INFO_BACKUP = pickle_data(READ, "CAMERA_INFO_PLANE.pickle", None)['CAMERA_INFO']
NEW_CAMERA_INFO_UP = pickle_data(READ, "NEW_CAMERA_INFO_1.pickle", None)['NEW_CAMERA_INFO']
NEW_CAMERA_INFO_UP_KEYS = list(NEW_CAMERA_INFO_UP.keys())
NEW_CAMERA_INFO_DOWN = pickle_data(READ, "NEW_CAMERA_INFO_-1.pickle", None)['NEW_CAMERA_INFO']
NEW_CAMERA_INFO_DOWN_KEYS = list(NEW_CAMERA_INFO_DOWN.keys())


BLOB_CNT = len(ORIGIN_DATA)
IMAGE_CNT = 120
DEGREE_CNT = 4
CAM_ID = 0
CAMERA_INFO = {}
BLOB_INFO = {}

if __name__ == "__main__":
    print('PTS')
    for i, leds in enumerate(ORIGIN_DATA):
        print(f"{np.array2string(leds, separator=', ')},")
    print('DIR')
    for i, dir in enumerate(DIRECTION):
        print(f"{np.array2string(dir, separator=', ')},")
        
    # show_calibrate_data(np.array(ORIGIN_DATA), np.array(DIRECTION))

    def filter_data(led_data, points2Ddata):
        LED_NUMBER = []
        points2D = []
        for i, blob_id in enumerate(led_data):
            if blob_id != -1:
                LED_NUMBER.append(blob_id)
                points2D.append(points2Ddata[i])
        points2D = np.array(np.array(points2D).reshape(len(points2D), -1), dtype=np.float64)
        return LED_NUMBER, points2D
    def pnp_solver(points2D_D, points3D):
        # print(points2D_D)
        ret_status = SUCCESS
        points2D_U = cv2.undistortPoints(points2D_D, camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][1])
        points2D_U = np.array(np.array(points2D_U).reshape(len(points2D_U), -1), dtype=np.float64)
        length = len(points2D_D)
        if length >= 5:
            METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC
        elif length == 4:
            METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P
        else:
            ret_status = ERROR

        if ret_status == SUCCESS:
            INPUT_ARRAY = [
                CAM_ID,
                points3D,
                points2D_U,
                default_cameraK,
                default_dist_coeffs
            ]
            _, rvec, tvec, _ = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
            # rvec이나 tvec가 None인 경우 continue
            if rvec is None or tvec is None:
                ret_status = ERROR
            
            # rvec이나 tvec에 nan이 포함된 경우 continue
            if np.isnan(rvec).any() or np.isnan(tvec).any():
                ret_status = ERROR

            # rvec과 tvec의 모든 요소가 0인 경우 continue
            if np.all(rvec == 0) and np.all(tvec == 0):
                ret_status = ERROR
            # print('rvec:', rvec)
            # print('tvec:', tvec)
        
        if ret_status == ERROR:
            return ERROR, points2D_U
        
        rvec_reshape = np.array(rvec).reshape(-1, 1)
        tvec_reshape = np.array(tvec).reshape(-1, 1)

        return (rvec_reshape, tvec_reshape), points2D_U

    def add_data(index, LED_NUMBER, DEGREE, points2D_D, points2D_U, points3D, RTVEC, frame_cnt, DO_BA, DO_CALIBRATION_TEST):
        if DO_BA == DONE:         
            BA_RT = pickle_data(READ, 'BA_RT.pickle', None)['BA_RT']
            ba_rvec = BA_RT[frame_cnt][:3]
            ba_tvec = BA_RT[frame_cnt][3:]
            ba_rvec_reshape = np.array(ba_rvec).reshape(-1, 1)
            ba_tvec_reshape = np.array(ba_tvec).reshape(-1, 1)
            # print('ba_rvec : ', ba_rvec)
            # print('ba_tvec : ', ba_tvec)
            # print('LED_NUMBER : ', LED_NUMBER)
            for blob_id in LED_NUMBER:
                BLOB_INFO[blob_id]['BA_RT']['rt']['rvec'].append(ba_rvec_reshape)
                BLOB_INFO[blob_id]['BA_RT']['rt']['tvec'].append(ba_tvec_reshape)
                BLOB_INFO[blob_id]['BA_RT']['status'].append(DONE)               
            CAMERA_INFO[f"{frame_cnt}"]['BA_RT']['rt']['rvec'] = ba_rvec_reshape
            CAMERA_INFO[f"{frame_cnt}"]['BA_RT']['rt']['tvec'] = ba_tvec_reshape          

        elif DO_CALIBRATION_TEST == DONE:
            RIGID_3D_TRANSFORM_PCA = pickle_data(READ, 'RIGID_3D_TRANSFORM.pickle', None)['PCA_ARRAY_LSM']
            RIGID_3D_TRANSFORM_IQR = pickle_data(READ, 'RIGID_3D_TRANSFORM.pickle', None)['IQR_ARRAY_LSM']
            points3D_PCA = []
            points3D_IQR = []
            for blob_id in LED_NUMBER:
                points3D_PCA.append(RIGID_3D_TRANSFORM_PCA[int(blob_id)])
                points3D_IQR.append(RIGID_3D_TRANSFORM_IQR[int(blob_id)])
            points3D_PCA = np.array(points3D_PCA, dtype=np.float64)
            points3D_IQR = np.array(points3D_IQR, dtype=np.float64)
            CAMERA_INFO[f"{frame_cnt}"]['points3D_PCA'] = points3D_PCA
            CAMERA_INFO[f"{frame_cnt}"]['points3D_IQR'] = points3D_IQR
        else:
            for idx, blob_id in enumerate(LED_NUMBER):
                BLOB_INFO[blob_id]['points2D_D']['greysum'].append(points2D_D[idx])
                BLOB_INFO[blob_id]['points2D_U']['greysum'].append(points2D_U[idx])
                if RTVEC != ERROR:
                    BLOB_INFO[blob_id]['OPENCV']['rt']['rvec'].append(RTVEC[0])
                    BLOB_INFO[blob_id]['OPENCV']['rt']['tvec'].append(RTVEC[1])
                    BLOB_INFO[blob_id]['OPENCV']['status'].append(DONE)
                else:
                    BLOB_INFO[blob_id]['OPENCV']['rt']['rvec'].append(NOT_SET)
                    BLOB_INFO[blob_id]['OPENCV']['rt']['tvec'].append(NOT_SET)
                    BLOB_INFO[blob_id]['OPENCV']['status'].append(NOT_SET)                    
            
            CAMERA_INFO[f"{frame_cnt}"]['points3D'] = points3D
            CAMERA_INFO[f"{frame_cnt}"]['points3D_origin'] = ORIGIN_DATA[list(LED_NUMBER), :]
            CAMERA_INFO[f"{frame_cnt}"]['points3D_legacy'] = MODEL_DATA[list(LED_NUMBER), :]
            CAMERA_INFO[f"{frame_cnt}"]['points2D']['greysum'] = points2D_D
            CAMERA_INFO[f"{frame_cnt}"]['points2D_U']['greysum'] = points2D_U            
            CAMERA_INFO[f"{frame_cnt}"]['LED_NUMBER'] =LED_NUMBER
            CAMERA_INFO[f"{frame_cnt}"]['ANGLE'] =DEGREE
            if RTVEC != ERROR:
                CAMERA_INFO[f"{frame_cnt}"]['OPENCV']['rt']['rvec'] = RTVEC[0]
                CAMERA_INFO[f"{frame_cnt}"]['OPENCV']['rt']['tvec'] = RTVEC[1]
                CAMERA_INFO[f"{frame_cnt}"]['OPENCV']['status'] = DONE
            # Set RT made by BLENDER
            if DEGREE == 0:
                camera_params = read_camera_log(os.path.join(script_dir,  f"./tmp/render/ARCTURAS/camera_log_0.txt"))
            elif DEGREE == 15:
                camera_params = read_camera_log(os.path.join(script_dir,  f"./tmp/render/ARCTURAS/camera_log_15.txt"))
            elif DEGREE == -15:
                camera_params = read_camera_log(os.path.join(script_dir,  f"./tmp/render/ARCTURAS/camera_log_-15.txt"))
            elif DEGREE == -30:
                camera_params = read_camera_log(os.path.join(script_dir,  f"./tmp/render/ARCTURAS/camera_log_-30.txt"))
            if camera_params != ERROR:
                brvec, btvec = camera_params[index + 1]
                brvec_reshape = np.array(brvec).reshape(-1, 1)
                btvec_reshape = np.array(btvec).reshape(-1, 1)
                # print('Blender rvec:', brvec_reshape.flatten(), ' tvec:', btvec_reshape.flatten())            
                for blob_id in LED_NUMBER:
                    BLOB_INFO[blob_id]['BLENDER']['rt']['rvec'].append(brvec_reshape)
                    BLOB_INFO[blob_id]['BLENDER']['rt']['tvec'].append(btvec_reshape)
                    BLOB_INFO[blob_id]['BLENDER']['status'].append(DONE)  
                CAMERA_INFO[f"{frame_cnt}"]['BLENDER']['rt']['rvec'] = brvec_reshape
                CAMERA_INFO[f"{frame_cnt}"]['BLENDER']['rt']['tvec'] = btvec_reshape   

    def gathering_data(DO_BA=NOT_SET, DO_CALIBRATION_TEST=NOT_SET):
        print('gathering_data START: ', DO_BA, ' ', DO_CALIBRATION_TEST)
        frame_cnt = 0
        for idx in range(IMAGE_CNT):
            # 0, -15 -30 15의 4가지 앵글에서 카메라뷰를 처리함
            # frame_cnt가 4배씩 증가
            # led가 3개이하로 보이면 예외 처리가 필요함 

            # 0 ##################            
            LED_NUMBER = CAMERA_INFO_BACKUP[f"{idx}"]['LED_NUMBER']
            points2D_D = CAMERA_INFO_BACKUP[f"{idx}"]['points2D']['greysum']
            points3D = CALIBRATION_DATA[list(LED_NUMBER), :]
            RET, points2D_U = pnp_solver(points2D_D, points3D)
            if RET != ERROR:
                if idx == 0:
                    print(f"0 frame_cnt {frame_cnt}")
                    print(np.array(RET[0]).flatten())
                    print(np.array(RET[1]).flatten())
            add_data(idx, LED_NUMBER, 0, points2D_D, points2D_U, points3D, RET, frame_cnt, DO_BA, DO_CALIBRATION_TEST)
            frame_cnt += 1
            ##################

            # -15 ##################            
            LED_NUMBER, points2D_D = filter_data(NEW_CAMERA_INFO_UP[NEW_CAMERA_INFO_UP_KEYS[idx*2]]['LED_NUMBER'], NEW_CAMERA_INFO_UP[NEW_CAMERA_INFO_UP_KEYS[idx*2]]['points2D'])
            points2D_D = np.array(np.array(points2D_D).reshape(len(points2D_D), -1), dtype=np.float64)
            points3D = CALIBRATION_DATA[list(LED_NUMBER), :]
            RET, points2D_U = pnp_solver(points2D_D, points3D)
            if RET != ERROR:
                if idx == 0:
                    print(f"-15 frame_cnt {frame_cnt}")
                    print(np.array(RET[0]).flatten())
                    print(np.array(RET[1]).flatten())
            add_data(idx, LED_NUMBER, -15, points2D_D, points2D_U, points3D, RET, frame_cnt, DO_BA, DO_CALIBRATION_TEST)
            frame_cnt += 1
            ##################

            # -30 ##################            
            LED_NUMBER, points2D_D = filter_data(NEW_CAMERA_INFO_UP[NEW_CAMERA_INFO_UP_KEYS[idx*2 + 1]]['LED_NUMBER'], NEW_CAMERA_INFO_UP[NEW_CAMERA_INFO_UP_KEYS[idx*2 + 1]]['points2D'])
            points2D_D = np.array(np.array(points2D_D).reshape(len(points2D_D), -1), dtype=np.float64)
            points3D = CALIBRATION_DATA[list(LED_NUMBER), :]
            RET, points2D_U = pnp_solver(points2D_D, points3D)
            if RET != ERROR:
                if idx == 0:
                    print(f"-30 frame_cnt {frame_cnt}")
                    print(np.array(RET[0]).flatten())
                    print(np.array(RET[1]).flatten())
            add_data(idx, LED_NUMBER, -30, points2D_D, points2D_U, points3D, RET, frame_cnt, DO_BA, DO_CALIBRATION_TEST)
            frame_cnt += 1
            ##################

            # 15 ##################            
            LED_NUMBER, points2D_D = filter_data(NEW_CAMERA_INFO_DOWN[NEW_CAMERA_INFO_DOWN_KEYS[idx]]['LED_NUMBER'], NEW_CAMERA_INFO_DOWN[NEW_CAMERA_INFO_DOWN_KEYS[idx]]['points2D'])   
            points2D_D = np.array(np.array(points2D_D).reshape(len(points2D_D), -1), dtype=np.float64)  
            points3D = CALIBRATION_DATA[list(LED_NUMBER), :]
            RET, points2D_U = pnp_solver(points2D_D, points3D)
            if RET != ERROR:
                if idx == 0:
                    print(f"15 frame_cnt {frame_cnt}")
                    print(np.array(RET[0]).flatten())
                    print(np.array(RET[1]).flatten())
            add_data(idx, LED_NUMBER, 15, points2D_D, points2D_U, points3D, RET, frame_cnt, DO_BA, DO_CALIBRATION_TEST)
            frame_cnt += 1
            ##################

        data = OrderedDict()
        data['BLOB_INFO'] = BLOB_INFO
        pickle_data(WRITE, 'BLOB_INFO.pickle', data)
        data = OrderedDict()
        data['CAMERA_INFO'] = CAMERA_INFO
        pickle_data(WRITE, 'CAMERA_INFO.pickle', data)
        print('data saved\n')

    def  Check_Calibration_data_combination(combination_cnt, **kwargs):
        print('Check_Calibration_data_combination START')
        info_name = kwargs.get('info_name')    
        CAMERA_INFO = pickle_data(READ, info_name, None)[info_name.split('.')[0]]
        RIGID_3D_TRANSFORM_IQR = pickle_data(READ, 'RIGID_3D_TRANSFORM.pickle', None)['IQR_ARRAY_LSM']

        print('Calibration cadidates')
        for blob_id, points_3d in enumerate(RIGID_3D_TRANSFORM_IQR):
            print(f"{points_3d},")    
        
        def STD_Analysis(points3D_data, label, combination):
            print(f"dataset:{points3D_data} combination_cnt:{combination}")
            frame_counts = []
            rvec_std_arr = []
            tvec_std_arr = []
            reproj_err_rates = []
            error_cnt = 0
            fail_reason = []

             # Use the default camera matrix as the intrinsic parameters
            intrinsics_single = torch.tensor(np.array([default_cameraK]), dtype=torch.float64)
            img = np.zeros((960, 1280, 3), dtype=np.uint8)
            for frame_cnt, cam_data in CAMERA_INFO.items(): 
                rvec_list = []
                tvec_list = []
                reproj_err_list = []
                
                # Prepare lists for batch processing
                batch_2d_points = []
                batch_3d_points = []
                
                LED_NUMBER = cam_data['LED_NUMBER']
                points3D = cam_data[points3D_data]
                points2D = cam_data['points2D']['greysum']
                points2D_U = cam_data['points2D_U']['greysum']
                # print('frame_cnt: ',frame_cnt)
                # print('points2D: ', len(points2D))
                # print('points2D_U: ', len(points2D_U))
                # print('points3D\n',points3D)
                LENGTH = len(LED_NUMBER)

                if LENGTH >= combination:
                    for comb in combinations(range(LENGTH), combination):
                        for perm in permutations(comb):
                            points3D_perm = points3D[list(perm), :]
                            points2D_perm = points2D[list(perm), :]
                            points2D_U_perm = points2D_U[list(perm), :]
                            if combination == 6:
                                if USING_KORNIA == DONE:
                                    # Using DLT
                                    # Kornia, Tensor
                                    # Add to batch lists
                                    batch_2d_points.append(points2D_U_perm)
                                    batch_3d_points.append(points3D_perm)
                                    continue
                                else:
                                    METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC          
                            elif combination == 5:
                                METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC
                            elif combination == 4:
                                METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P
                            else:
                                continue

                            INPUT_ARRAY = [
                                CAM_ID,
                                points3D_perm,
                                points2D_U_perm,
                                default_cameraK,
                                default_dist_coeffs
                            ]
                            _, rvec, tvec, _ = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
                            # rvec이나 tvec가 None인 경우 continue
                            if rvec is None or tvec is None:
                                continue                            
                            # rvec이나 tvec에 nan이 포함된 경우 continue
                            if np.isnan(rvec).any() or np.isnan(tvec).any():
                                continue
                            # rvec과 tvec의 모든 요소가 0인 경우 continue
                            if np.all(rvec == 0) and np.all(tvec == 0):
                                continue
                            # print('rvec:', rvec)
                            # print('tvec:', tvec)

                            RER, reproj_2d = reprojection_error(points3D_perm,
                                                     points2D_perm,
                                                     rvec, tvec,
                                                     camera_matrix[CAM_ID][0],
                                                     camera_matrix[CAM_ID][1])
                            if int(frame_cnt) == 10:
                                # print(reproj_2d)
                                for (x, y) in reproj_2d:
                                    cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)

                            
                            if RER > 2.5:
                                error_cnt+=1
                                # print('points3D_perm ', points3D_perm)
                                # print('points3D_data ', points3D_data)
                                # print('list(perm): ', list(perm), ' RER: ', RER)
                                fail_reason.append([points3D_data, error_cnt, list(perm), RER, rvec.flatten(), tvec.flatten()])
                            else:
                                rvec_list.append(np.linalg.norm(rvec))
                                tvec_list.append(np.linalg.norm(tvec))                      
                                reproj_err_list.append(RER)                                
                    
                    if USING_KORNIA == DONE:
                        # Convert lists of numpy arrays to single numpy arrays
                        batch_2d_points_np = np.array(batch_2d_points)
                        batch_3d_points_np = np.array(batch_3d_points)
                        # Convert numpy arrays to PyTorch tensors
                        batch_2d_points = torch.from_numpy(batch_2d_points_np)
                        batch_3d_points = torch.from_numpy(batch_3d_points_np)
                        # Duplicate intrinsics for each item in the batch
                        intrinsics = intrinsics_single.repeat(batch_2d_points.shape[0], 1, 1)
                        # Use tensors in kornia function
                        pred_world_to_cam = K.geometry.solve_pnp_dlt(batch_3d_points, batch_2d_points, intrinsics)
                        # For each predicted world_to_cam matrix...
                        for i in range(pred_world_to_cam.shape[0]):
                            # Unpack the rotation and translation vectors from the world_to_cam matrix
                            # print(pred_world_to_cam[i, :3, :3])
                            rvec = cv2.Rodrigues(pred_world_to_cam[i, :3, :3].cpu().numpy())[0]
                            tvec = pred_world_to_cam[i, :3, 3].cpu().numpy()
                            if rvec is None or tvec is None:
                                continue                            
                            # rvec이나 tvec에 nan이 포함된 경우 continue
                            if np.isnan(rvec).any() or np.isnan(tvec).any():
                                continue
                            # rvec과 tvec의 모든 요소가 0인 경우 continue
                            if np.all(rvec == 0) and np.all(tvec == 0):
                                continue
                            # Project the 3D points to the image plane using cv2.projectPoints
                            projected_2d_points, _ = cv2.projectPoints(batch_3d_points[i].cpu().numpy(), rvec, tvec, default_cameraK, default_dist_coeffs)
                            # Compute the reprojection error
                            RER = np.sum((projected_2d_points.squeeze() - batch_2d_points[i].cpu().numpy())**2)
                            if RER > 2.5:
                                error_cnt+=1
                                # print('points3D_perm ', points3D_perm)
                                # print('points3D_data ', points3D_data)
                                # print('list(perm): ', list(perm), ' RER: ', RER)
                                fail_reason.append([points3D_data, error_cnt, list(perm), RER, rvec.flatten(), tvec.flatten()])
                            else:
                                rvec_list.append(np.linalg.norm(rvec))
                                tvec_list.append(np.linalg.norm(tvec))                      
                                reproj_err_list.append(RER)                            
                        
                    if rvec_list and tvec_list and reproj_err_list:
                        frame_counts.append(frame_cnt)
                        rvec_std = np.std(rvec_list)
                        tvec_std = np.std(tvec_list)
                        reproj_err_rate = np.mean(reproj_err_list)

                        rvec_std_arr.append(rvec_std)
                        tvec_std_arr.append(tvec_std)
                        reproj_err_rates.append(reproj_err_rate)
                        
            cv2.imwrite(f"{points3D_data}_ouput.png", img)
            return frame_counts, rvec_std_arr, tvec_std_arr, reproj_err_rates, label, fail_reason
        
        all_data = []
        points3D_datas = ['points3D_origin', 'points3D_legacy', 'points3D_IQR']
        colors = ['r', 'g', 'b']
        points3D_LENGTH = len(points3D_datas)
        for COMBINATION in combination_cnt:
            fig, axs = plt.subplots(3, 1, figsize=(15, 15))
            fig.suptitle(f'Calibration Data Analysis for Combination: {COMBINATION}')  # Set overall title


            summary_text = ""

            for idx, points3D_data in enumerate(points3D_datas):
                frame_counts, rvec_std_arr, tvec_std_arr, reproj_err_rates, label, fail_reason = STD_Analysis(points3D_data, points3D_datas, COMBINATION)

                axs[0].plot(frame_counts, rvec_std_arr, colors[idx]+'-', label=f'rvec std {label[idx]}', alpha=0.5)
                axs[0].plot(frame_counts, tvec_std_arr, colors[idx]+'--', label=f'tvec std {label[idx]}', alpha=0.5)
                axs[1].plot(frame_counts, reproj_err_rates, colors[idx], label=f'Reprojection error rate {label[idx]}', alpha=0.5)
                
                # Calculate and store the average and standard deviation for each data set
                avg_rvec_std = np.mean(rvec_std_arr)
                std_rvec_std = np.std(rvec_std_arr)
                avg_tvec_std = np.mean(tvec_std_arr)
                std_tvec_std = np.std(tvec_std_arr)
                avg_reproj_err = np.mean(reproj_err_rates)
                std_reproj_err = np.std(reproj_err_rates)
                
                summary_text += f"== {label} ==\n"
                summary_text += f"Rvec Std: Mean = {avg_rvec_std:.6f}, Std = {std_rvec_std:.6f}\n"
                summary_text += f"Tvec Std: Mean = {avg_tvec_std:.6f}, Std = {std_tvec_std:.6f}\n"
                summary_text += f"Reproj Err: Mean = {avg_reproj_err:.6f}, Std = {std_reproj_err:.6f}\n"
                # summary_text += f"error cnt (over RER 2px) {error_cnt:.6f}\n"
                summary_text += "\n"
                all_data.append([label, COMBINATION, avg_rvec_std, std_rvec_std, avg_tvec_std, std_tvec_std, avg_reproj_err, std_reproj_err])  # Store data for all combinations

            axs[0].legend()
            axs[0].set_xlabel('frame_cnt')
            axs[0].set_ylabel('std')
            axs[0].set_title('Standard Deviation of rvec and tvec Magnitude per Frame')

            axs[1].legend()
            axs[1].set_xlabel('frame_cnt')
            axs[1].set_ylabel('error rate')
            axs[1].set_title('Mean Reprojection Error Rate per Frame')
            
            axs[2].axis('off')  # Hide axes for the text plot
            axs[2].text(0, 0, summary_text, fontsize=10)

            # Reducing the number of X-ticks to avoid crowding
            for ax in axs[:2]:
                ax.set_xticks(ax.get_xticks()[::5])

            plt.subplots_adjust(hspace=0.5)  # Add space between subplots
        original_labels = [f"{item[0]} {item[1]}" for item in all_data] # Here, we first create original labels
        labels = combination_cnt * (len(original_labels) // len(combination_cnt)) # Now we can use len(original_labels)

        avg_rvec_std_values = [item[2] for item in all_data]
        std_rvec_std_values = [item[3] for item in all_data]
        avg_tvec_std_values = [item[4] for item in all_data]
        std_tvec_std_values = [item[5] for item in all_data]
        avg_reproj_err_values = [item[6] for item in all_data]
        std_reproj_err_values = [item[7] for item in all_data]

        x = np.arange(len(labels) // points3D_LENGTH)
        print(x)
        print(labels)
        print(labels[:len(combination_cnt)])
        width = 0.35
        fig, axs = plt.subplots(4, 1, figsize=(15, 30)) # increase the figure size

        # Rvec subplot
        rects1 = axs[0].bar(x - width / points3D_LENGTH*2, avg_rvec_std_values[::3], width / points3D_LENGTH, color='r', label='Avg Rvec Std for points3D_ORIGIN')
        rects2 = axs[0].bar(x  / points3D_LENGTH*2, avg_rvec_std_values[1::3], width / points3D_LENGTH, color='g', label='Avg Rvec Std for points3D_LEGACY')
        rects3 = axs[0].bar(x + width / points3D_LENGTH*2, avg_rvec_std_values[2::3], width / points3D_LENGTH, color='b', label='Avg Rvec Std for points3D_IQR')
        

        axs[0].set_xlabel('Combination')
        axs[0].set_ylabel('Values')
        axs[0].set_title('Average Rvec Standard Deviations for All Combinations')
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(labels[:len(combination_cnt)])
        axs[0].legend()

        # Tvec subplot
        rects4 = axs[1].bar(x - width / points3D_LENGTH*2, avg_tvec_std_values[::3], width / points3D_LENGTH, color='r', label='Avg Tvec Std for points3D_ORIGIN')
        rects5 = axs[1].bar(x  / points3D_LENGTH*2, avg_tvec_std_values[1::3], width / points3D_LENGTH, color='g', label='Avg Tvec Std for points3D_LEGACY')
        rects6 = axs[1].bar(x + width / points3D_LENGTH*2, avg_tvec_std_values[2::3], width / points3D_LENGTH, color='b', label='Avg Tvec Std for points3D_IQR')

        axs[1].set_xlabel('Combination')
        axs[1].set_ylabel('Values')
        axs[1].set_title('Average Tvec Standard Deviations for All Combinations')
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(labels[:len(combination_cnt)])
        axs[1].legend()

        # Reproj_err subplot
        rects7 = axs[2].bar(x - width / points3D_LENGTH*2, avg_reproj_err_values[::3], width / points3D_LENGTH, color='r', label='Avg Reproj Err Std for points3D_ORIGIN')
        rects8 = axs[2].bar(x  / points3D_LENGTH*2, avg_reproj_err_values[1::3], width / points3D_LENGTH, color='g', label='Avg Reproj Err Std for points3D_LEGACY')
        rects9 = axs[2].bar(x + width / points3D_LENGTH*2, avg_reproj_err_values[2::3], width / points3D_LENGTH, color='b', label='Avg Reproj Err Std for points3D_IQR')

        axs[2].set_xlabel('Combination')
        axs[2].set_ylabel('Values')
        axs[2].set_title('Average Reprojection Error for All Combinations')
        axs[2].set_xticks(x)
        axs[2].set_xticklabels(labels[:len(combination_cnt)])
        axs[2].legend()

        # Organize data into a dictionary with column labels as keys
        summary_data = {
            'LEDCount:': labels[:len(combination_cnt)],
            'AvgRvecStd(points3D_ORIGIN):': avg_rvec_std_values[::3],
            'AvgRvecStd(points3D_LEGACY):': avg_rvec_std_values[1::3],
            'AvgRvecStd(points3D_IQR):': avg_rvec_std_values[2::3],

            'AvgTvecStd(points3D_ORIGIN):': avg_tvec_std_values[::3],
            'AvgTvecStd(points3D_LEGACY):': avg_tvec_std_values[1::3],
            'AvgTvecStd(points3D_IQR):': avg_tvec_std_values[2::3],

            'AvgReprojErr(points3D_ORIGIN):': avg_reproj_err_values[::3],
            'AvgReprojErr(points3D_LEGACY):': avg_reproj_err_values[1::3],
            'AvgReprojErr(points3D_IQR):': avg_reproj_err_values[2::3]
        }

        # Create DataFrame from dictionary
        df = pd.DataFrame(summary_data)
        # 'LEDCount:' 열을 제외한 새로운 DataFrame 생성
        df_without_ledcount = df.drop('LEDCount:', axis=1)

        # 통계를 가져옵니다
        desc_stats = df_without_ledcount.describe()

        # # Get descriptive statistics
        # desc_stats = df.describe()

        # Convert to string and add to summary text
        summary_text = df.to_string(index=False)
        summary_text += "\n\nDescriptive Statistics:\n" + desc_stats.to_string()

        # Remove existing text subplot
        fig.delaxes(axs[3])

        # Add table as text subplot
        axs[3] = fig.add_subplot(414)
        axs[3].axis('off')
        axs[3].text(0.5, 0.5, summary_text, ha='center', va='center')

        # Print to console
        print(summary_text)

        # Write to log file
        with open('summary.log', 'w') as f:
            f.write(summary_text)

        file = 'FAIL_REASON.pickle'
        data = OrderedDict()
        data['FAIL_REASON'] = fail_reason
        ret = pickle_data(WRITE, file, data)
        if ret != ERROR:
            print('data saved')
        
        plt.subplots_adjust(hspace=0.5)


    ORIGIN_DATA = np.array(ORIGIN_DATA)
    MODEL_DATA = np.array(MODEL_DATA)
    # MAIN START
    for blob_id in range(BLOB_CNT):
        BLOB_INFO[blob_id] = copy.deepcopy(BLOB_INFO_STRUCTURE)
    
    for idx in range(IMAGE_CNT * DEGREE_CNT):
        # 0, -15 -30 15의 4가지 앵글에서 카메라뷰를 처리함
        # frame_cnt가 4배씩 증가
        CAMERA_INFO[f"{idx}"] = copy.deepcopy(CAMERA_INFO_STRUCTURE)
    
    
    from Advanced_Calibration import BA_RT, remake_3d_for_blob_info, LSM, init_plot, draw_result

    ax1, ax2 = init_plot(ORIGIN_DATA)

    # Phase 1
    gathering_data()

    # Phase 2
    BA_RT(info_name='CAMERA_INFO.pickle', save_to='BA_RT.pickle', target='BLENDER') 
    gathering_data(DO_BA=DONE)
    remake_3d_for_blob_info(blob_cnt=BLOB_CNT, info_name='BLOB_INFO.pickle', undistort=DONE, opencv=DONE, blender=DONE, ba_rt=DONE)
    LSM(TARGET_DEVICE, ORIGIN_DATA, info_name='REMADE_3D_INFO_BA')

    # Phase 3
    gathering_data(DO_CALIBRATION_TEST=DONE)
    draw_result(ORIGIN_DATA, ax1=ax1, ax2=ax2, opencv=DONE, blender=DONE, ba_rt=DONE) 

    # TEST
    USING_KORNIA = NOT_SET
    combination_cnt = [6]
    Check_Calibration_data_combination(combination_cnt, info_name='CAMERA_INFO.pickle')

    plt.show()

    print('\n\n')
    print('########## DONE ##########')

    