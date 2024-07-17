
from Advanced_Function import *  # 필요한 함수와 클래스를 임포트하십시오.

import multiprocessing as mp
import psutil

# Your existing functions and imports here
USING_KORNIA = NOT_SET
CAM_ID = 0

def STD_Analysis(points3D_data, combination, CAMERA_INFO):
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
        # print('points2D: ', points2D)
        # print('points2D_U: ', points2D_U)
        # print('points3D\n', points3D)
        # print('LED_NUMBER:', LED_NUMBER)
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
    return frame_counts, rvec_std_arr, tvec_std_arr, reproj_err_rates, fail_reason


def log_process_core_info():
    pid = os.getpid()
    process = psutil.Process(pid)
    cpu_affinity = process.cpu_affinity()
    cpu_num = psutil.Process(pid).cpu_num()
    print(f"Process {pid} is running on CPU {cpu_num} with affinity {cpu_affinity}")

def process_points3D(points3D_data, combination_cnt, CAMERA_INFO):
    log_process_core_info()  # Log core info
    all_data = []
    fail_reasons = []
    
    for COMBINATION in combination_cnt:
        frame_counts, rvec_std_arr, tvec_std_arr, reproj_err_rates, fail_reason = STD_Analysis(points3D_data, COMBINATION, CAMERA_INFO)
        
        avg_rvec_std = np.mean(rvec_std_arr)
        std_rvec_std = np.std(rvec_std_arr)
        avg_tvec_std = np.mean(tvec_std_arr)
        std_tvec_std = np.std(tvec_std_arr)
        avg_reproj_err = np.mean(reproj_err_rates)
        std_reproj_err = np.std(reproj_err_rates)
        
        all_data.append([points3D_data, COMBINATION, avg_rvec_std, std_rvec_std, avg_tvec_std, std_tvec_std, avg_reproj_err, std_reproj_err])
        fail_reasons.extend(fail_reason)
    
    summary_data = {
        'Points3DData': [item[0] for item in all_data],
        'Combination': [item[1] for item in all_data],
        'AvgRvecStd': [item[2] for item in all_data],
        'StdRvecStd': [item[3] for item in all_data],
        'AvgTvecStd': [item[4] for item in all_data],
        'StdTvecStd': [item[5] for item in all_data],
        'AvgReprojErr': [item[6] for item in all_data],
        'StdReprojErr': [item[7] for item in all_data]
    }
    
    df = pd.DataFrame(summary_data)
    df.to_csv(f'summary_{points3D_data}.csv', index=False)
    
    file = f'FAIL_REASON_{points3D_data}.pickle'
    data = OrderedDict()
    data['FAIL_REASON'] = fail_reasons
    ret = pickle_data(WRITE, file, data)
    if ret != ERROR:
        print('data saved')


def Check_Calibration_data_combination(combination_cnt, **kwargs):
    print('Check_Calibration_data_combination START')
    info_name = kwargs.get('info_name')
    CAMERA_INFO = pickle_data(READ, info_name, None)[info_name.split('.')[0]]
    RIGID_3D_TRANSFORM_IQR = pickle_data(READ, 'RIGID_3D_TRANSFORM.pickle', None)['IQR_ARRAY_LSM']

    print('Calibration cadidates')
    for blob_id, points_3d in enumerate(RIGID_3D_TRANSFORM_IQR):
        print(f"{points_3d},")    
        
    points3D_datas = ['points3D_origin', 'points3D_legacy', 'points3D_IQR']

    processes = []
    for points3D_data in points3D_datas:
        p = mp.Process(target=process_points3D, args=(points3D_data, combination_cnt, CAMERA_INFO))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    
    # Combine results from all processes
    combined_data = []
    for points3D_data in points3D_datas:
        try:
            df = pd.read_csv(f'summary_{points3D_data}.csv')
            combined_data.append(df)
        except FileNotFoundError:
            print(f'File summary_{points3D_data}.csv not found')
    
    if combined_data:
        combined_df = pd.concat(combined_data, axis=0)
        combined_df.to_csv('combined_summary.csv', index=False)
        
        # Visualization based on combination_cnt
        fig, axs = plt.subplots(len(combination_cnt) * 3, 1, figsize=(15, 10 * len(combination_cnt)))
        axs = axs.flatten()
        
        index = 0
        for combination in combination_cnt:
            subset = combined_df[combined_df['Combination'] == combination]
            labels = subset['Points3DData']
            x = np.arange(len(labels))
            width = 0.25
            
            # Rvec Std plot for current combination
            axs[index].bar(x - width/2, subset['AvgRvecStd'], width, label='Avg Rvec Std')
            axs[index].bar(x + width/2, subset['StdRvecStd'], width, label='Std Rvec Std')
            axs[index].set_xlabel('Points3DData')
            axs[index].set_ylabel('Values')
            axs[index].set_title(f'Rvec Standard Deviations (Combination {combination})')
            axs[index].set_xticks(x)
            axs[index].set_xticklabels(labels)
            axs[index].legend()
            index += 1
            
            # Tvec Std plot for current combination
            axs[index].bar(x - width/2, subset['AvgTvecStd'], width, label='Avg Tvec Std')
            axs[index].bar(x + width/2, subset['StdTvecStd'], width, label='Std Tvec Std')
            axs[index].set_xlabel('Points3DData')
            axs[index].set_ylabel('Values')
            axs[index].set_title(f'Tvec Standard Deviations (Combination {combination})')
            axs[index].set_xticks(x)
            axs[index].set_xticklabels(labels)
            axs[index].legend()
            index += 1
            
            # Reproj Err plot for current combination
            axs[index].bar(x - width/2, subset['AvgReprojErr'], width, label='Avg Reproj Err')
            axs[index].bar(x + width/2, subset['StdReprojErr'], width, label='Std Reproj Err')
            axs[index].set_xlabel('Points3DData')
            axs[index].set_ylabel('Values')
            axs[index].set_title(f'Reprojection Error (Combination {combination})')
            axs[index].set_xticks(x)
            axs[index].set_xticklabels(labels)
            axs[index].legend()
            index += 1
        
        summary_text = combined_df.to_string(index=False)
        desc_stats = combined_df.describe()
        summary_text += "\n\nDescriptive Statistics:\n" + desc_stats.to_string()
        
        plt.subplots_adjust(hspace=0.5)
        plt.savefig('combined_output.png')
        
        # Print and save summary
        print(summary_text)
        with open('combined_summary.log', 'w') as f:
            f.write(summary_text)
        plt.close()
    else:
        print("No data to combine and visualize.")

if __name__ == "__main__":
    combination_cnt = [4]
    
    start_time = time.time()
    Check_Calibration_data_combination(combination_cnt, info_name='CAMERA_INFO.pickle')
    end_time = time.time()
    elapsed_time_ms = (end_time - start_time)
    print(f"Processing time: {elapsed_time_ms:.2f} s")
    
    plt.show()
