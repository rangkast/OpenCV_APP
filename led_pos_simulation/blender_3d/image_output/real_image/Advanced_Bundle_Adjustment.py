from Advanced_Function import *

ANGLE = 3

CALIBRATION_DATA = np.array([
[-0.00528268, -0.03654436, 0.00445737],
[0.00975209, -0.04726801, 0.00416382],
[0.02991425, -0.05128519, 0.00376553],
[0.05157105, -0.04543511, 0.00313119],
[0.06950148, -0.0294263, 0.00313804],
[0.07726291, -0.01311312, 0.00290795],
[0.07830993, 0.00950335, 0.00290008],
[0.0717, 0.0261568, 0.00298388],
[0.05318036, 0.04458388, 0.00361513],
[0.03439977, 0.05093834, 0.00360017],
[0.01258188, 0.04842228, 0.00427163],
[-0.00531589, 0.03654443, 0.00443206],
[-0.01719296, -0.02498431, 0.01686793],
[-0.00735957, -0.03637139, 0.01720165],
[0.02462765, -0.05091951, 0.01816547],
[0.04458448, -0.04728218, 0.01847828],
[0.06122704, -0.03575422, 0.01903943],
[0.07374303, -0.01471812, 0.01925705],
[0.07167901, 0.02046464, 0.01915338],
[0.05465228, 0.04161754, 0.01860349],
[0.03432296, 0.05027539, 0.0182099],
[0.01237024, 0.04873717, 0.01758825],
[-0.00741192, 0.03632198, 0.01724611],
[-0.01722993, 0.02485586, 0.01684302],
])


# ToDo

TARGET_DEVICE = 'ARCTURAS'
MODEL_DATA, DIRECTION = init_coord_json(f"{script_dir}/jsons/specs/arcturas_right.json")
CAMERA_INFO_BACKUP = pickle_data(READ, "CAMERA_INFO_BACKUP.pickle", None)['CAMERA_INFO']
NEW_CAMERA_INFO_UP = pickle_data(READ, "NEW_CAMERA_INFO_1.pickle", None)['NEW_CAMERA_INFO']
NEW_CAMERA_INFO_UP_KEYS = list(NEW_CAMERA_INFO_UP.keys())
NEW_CAMERA_INFO_DOWN = pickle_data(READ, "NEW_CAMERA_INFO_-1.pickle", None)['NEW_CAMERA_INFO']
NEW_CAMERA_INFO_DOWN_KEYS = list(NEW_CAMERA_INFO_DOWN.keys())

# CALIBRATION_DATA = np.array(MODEL_DATA)

BLOB_CNT = len(MODEL_DATA)
IMAGE_CNT = 120
DEGREE_CNT = 4
CAM_ID = 0
CAMERA_INFO = {}
BLOB_INFO = {}

if __name__ == "__main__":
    print('PTS')
    for i, leds in enumerate(MODEL_DATA):
        print(f"{np.array2string(leds, separator=', ')},")
    print('DIR')
    for i, dir in enumerate(DIRECTION):
        print(f"{np.array2string(dir, separator=', ')},")
        
    # show_calibrate_data(np.array(MODEL_DATA), np.array(DIRECTION))

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
            CAMERA_INFO[f"{frame_cnt}"]['points3D_origin'] = MODEL_DATA[list(LED_NUMBER), :]
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

    def Check_Calibration_data_combination(combination_cnt, **kwargs):
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
            for frame_cnt, cam_data in CAMERA_INFO.items():       
                
                rvec_list = []
                tvec_list = []
                reproj_err_list = []

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
                            if combination >= 5:
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

                            RER = reprojection_error(points3D_perm,
                                                    points2D_perm,
                                                    rvec, tvec,
                                                    camera_matrix[CAM_ID][0],
                                                    camera_matrix[CAM_ID][1])
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


            return frame_counts, rvec_std_arr, tvec_std_arr, reproj_err_rates, label, fail_reason
        
        all_data = []
        for COMBINATION in combination_cnt:
            fig, axs = plt.subplots(3, 1, figsize=(15, 15))
            fig.suptitle(f'Calibration Data Analysis for Combination: {COMBINATION}')  # Set overall title

            points3D_datas = ['points3D_origin', 'points3D_IQR']
            colors = ['r', 'b']
            summary_text = ""

            for idx, points3D_data in enumerate(points3D_datas):
                frame_counts, rvec_std_arr, tvec_std_arr, reproj_err_rates, label, fail_reason = STD_Analysis(points3D_data, points3D_data, COMBINATION)

                axs[0].plot(frame_counts, rvec_std_arr, colors[idx]+'-', label=f'rvec std {label}', alpha=0.5)
                axs[0].plot(frame_counts, tvec_std_arr, colors[idx]+'--', label=f'tvec std {label}', alpha=0.5)
                axs[1].plot(frame_counts, reproj_err_rates, colors[idx], label=f'Reprojection error rate {label}', alpha=0.5)
                
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

        x = np.arange(len(labels) // 2)
        print(x)
        print(labels)
        print(labels[:len(combination_cnt)])
        width = 0.35
        fig, axs = plt.subplots(4, 1, figsize=(15, 30)) # increase the figure size

        # Rvec subplot
        rects1 = axs[0].bar(x - width / 4, avg_rvec_std_values[::2], width / 2, color='r', label='Avg Rvec Std for points3D')
        rects2 = axs[0].bar(x + width / 4, avg_rvec_std_values[1::2], width / 2, color='b', label='Avg Rvec Std for points3D_IQR')

        axs[0].set_xlabel('Combination')
        axs[0].set_ylabel('Values')
        axs[0].set_title('Average Rvec Standard Deviations for All Combinations')
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(labels[:len(combination_cnt)])
        axs[0].legend()

        # Tvec subplot
        rects3 = axs[1].bar(x - width / 4, avg_tvec_std_values[::2], width / 2, color='r', label='Avg Tvec Std for points3D')
        rects4 = axs[1].bar(x + width / 4, avg_tvec_std_values[1::2], width / 2, color='b', label='Avg Tvec Std for points3D_IQR')

        axs[1].set_xlabel('Combination')
        axs[1].set_ylabel('Values')
        axs[1].set_title('Average Tvec Standard Deviations for All Combinations')
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(labels[:len(combination_cnt)])
        axs[1].legend()

        # Reproj_err subplot
        rects5 = axs[2].bar(x - width / 4, avg_reproj_err_values[::2], width / 2, color='r', label='Avg Reproj Err for points3D')
        rects6 = axs[2].bar(x + width / 4, avg_reproj_err_values[1::2], width / 2, color='b', label='Avg Reproj Err for points3D_IQR')

        axs[2].set_xlabel('Combination')
        axs[2].set_ylabel('Values')
        axs[2].set_title('Average Reprojection Error for All Combinations')
        axs[2].set_xticks(x)
        axs[2].set_xticklabels(labels[:len(combination_cnt)])
        axs[2].legend()

        # Organize data into a dictionary with column labels as keys
        summary_data = {
            'LEDCount:': labels[:len(combination_cnt)],
            'AvgRvecStd(points3D):': avg_rvec_std_values[::2],
            'AvgRvecStd(points3D_IQR):': avg_rvec_std_values[1::2],
            'AvgTvecStd(points3D):': avg_tvec_std_values[::2],
            'AvgTvecStd(points3D_IQR):': avg_tvec_std_values[1::2],
            'AvgReprojErr(points3D):': avg_reproj_err_values[::2],
            'AvgReprojErr(points3D_IQR):': avg_reproj_err_values[1::2]
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


    MODEL_DATA = np.array(MODEL_DATA)
    # MAIN START
    for blob_id in range(BLOB_CNT):
        BLOB_INFO[blob_id] = copy.deepcopy(BLOB_INFO_STRUCTURE)
    
    for idx in range(IMAGE_CNT * DEGREE_CNT):
        # 0, -15 -30 15의 4가지 앵글에서 카메라뷰를 처리함
        # frame_cnt가 4배씩 증가
        CAMERA_INFO[f"{idx}"] = copy.deepcopy(CAMERA_INFO_STRUCTURE)
    
    
    from Advanced_Calibration import BA_RT, remake_3d_for_blob_info, LSM, init_plot, draw_result

    ax1, ax2 = init_plot(MODEL_DATA)

    # Phase 1
    gathering_data()


    # Phase 2
    BA_RT(info_name='CAMERA_INFO.pickle', save_to='BA_RT.pickle', target='BLENDER') 
    gathering_data(DO_BA=DONE)
    remake_3d_for_blob_info(blob_cnt=BLOB_CNT, info_name='BLOB_INFO.pickle', undistort=DONE, opencv=DONE, blender=DONE, ba_rt=DONE)
    LSM(TARGET_DEVICE, MODEL_DATA, info_name='REMADE_3D_INFO_BA')

    # Phase 3
    gathering_data(DO_CALIBRATION_TEST=DONE)
    draw_result(MODEL_DATA, ax1=ax1, ax2=ax2, opencv=DONE, blender=DONE, ba_rt=DONE) 

    # TEST
    combination_cnt = [4]
    Check_Calibration_data_combination(combination_cnt, info_name='CAMERA_INFO.pickle')

    plt.show()

    print('\n\n')
    print('########## DONE ##########')

    