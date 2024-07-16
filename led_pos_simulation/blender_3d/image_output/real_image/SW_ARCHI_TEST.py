from Advanced_Function import *

import torch
import torch.multiprocessing as mp
from itertools import combinations, permutations
import cv2
import numpy as np

def reprojection_error(points3D, points2D, rvec, tvec, camera_matrix, dist_coeffs):
    projected_points, _ = cv2.projectPoints(points3D, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.squeeze()
    error = np.sqrt(np.sum((points2D - projected_points)**2, axis=1))
    return np.mean(error), projected_points

def process_combination(gpu_id, combinations_data, result_queue):
    torch.cuda.set_device(gpu_id)
    for comb, perm, points3D, points2D_U, points2D, USING_KORNIA, intrinsics_single, default_cameraK, default_dist_coeffs, camera_matrix, CAM_ID in combinations_data:
        points3D_perm = points3D[list(perm), :].cuda()
        points2D_U_perm = points2D_U[list(perm), :].cuda()
        if USING_KORNIA == DONE:
            result_queue.put((points3D_perm.cpu().numpy(), points2D_U_perm.cpu().numpy(), None, None))
        else:
            METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC
            INPUT_ARRAY = [CAM_ID, points3D_perm.cpu().numpy(), points2D_U_perm.cpu().numpy(), default_cameraK, default_dist_coeffs]
            _, rvec, tvec, _ = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
            result_queue.put((points3D_perm.cpu().numpy(), points2D_U_perm.cpu().numpy(), rvec, tvec))

def STD_Analysis(points3D_data, label, combination, CAMERA_INFO, USING_KORNIA, default_cameraK, default_dist_coeffs, camera_matrix, CAM_ID):
    print(f"dataset:{points3D_data} combination_cnt:{combination}")
    frame_counts = []
    rvec_std_arr = []
    tvec_std_arr = []
    reproj_err_rates = []
    error_cnt = 0
    fail_reason = []

    intrinsics_single = torch.tensor(np.array([default_cameraK]), dtype=torch.float64).cuda()
    img = np.zeros((960, 1280, 3), dtype=np.uint8)
    result_queue = mp.Queue()
    processes = []

    for frame_cnt, cam_data in CAMERA_INFO.items():
        rvec_list = []
        tvec_list = []
        reproj_err_list = []

        LED_NUMBER = cam_data['LED_NUMBER']
        points3D = torch.tensor(cam_data[points3D_data], dtype=torch.float64).cuda()
        points2D = cam_data['points2D']['greysum']
        points2D_U = cam_data['points2D_U']['greysum']
        LENGTH = len(LED_NUMBER)

        if LENGTH >= combination:
            combinations_data = [(comb, perm, points3D, points2D_U, points2D, USING_KORNIA, intrinsics_single, default_cameraK, default_dist_coeffs, camera_matrix, CAM_ID)
                                 for comb in combinations(range(LENGTH), combination)
                                 for perm in permutations(comb)]

            # Split the data among available GPUs
            num_gpus = torch.cuda.device_count()
            chunk_size = len(combinations_data) // num_gpus
            for i in range(num_gpus):
                chunk = combinations_data[i * chunk_size:(i + 1) * chunk_size]
                p = mp.Process(target=process_combination, args=(i, chunk, result_queue))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            while not result_queue.empty():
                points3D_perm, points2D_U_perm, rvec, tvec = result_queue.get()
                if USING_KORNIA == DONE and rvec is None and tvec is None:
                    continue
                elif rvec is None or tvec is None or np.isnan(rvec).any() or np.isnan(tvec).any() or np.all(rvec == 0) and np.all(tvec == 0):
                    continue

                RER, reproj_2d = reprojection_error(points3D_perm, points2D_U_perm, rvec, tvec, camera_matrix[CAM_ID][0], camera_matrix[CAM_ID][1])
                if int(frame_cnt) == 10:
                    for (x, y) in reproj_2d:
                        cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)

                if RER > 2.5:
                    error_cnt += 1
                    fail_reason.append([points3D_data, error_cnt, list(perm), RER, rvec.flatten(), tvec.flatten()])
                else:
                    rvec_list.append(np.linalg.norm(rvec))
                    tvec_list.append(np.linalg.norm(tvec))
                    reproj_err_list.append(RER)

            if rvec_list and tvec_list and reproj_err_list:
                frame_counts.append(frame_cnt)
                rvec_std_arr.append(np.std(rvec_list))
                tvec_std_arr.append(np.std(tvec_list))
                reproj_err_rates.append(np.mean(reproj_err_list))

    cv2.imwrite(f"{points3D_data}_output.png", img)
    return frame_counts, rvec_std_arr, tvec_std_arr, reproj_err_rates, label, fail_reason

def new_Check_Calibration_data_combination(combination_cnt, info_name, USING_KORNIA, default_cameraK, default_dist_coeffs, camera_matrix, CAM_ID):
    print('Check_Calibration_data_combination START')
    CAMERA_INFO = pickle_data(READ, info_name, None)[info_name.split('.')[0]]
    RIGID_3D_TRANSFORM_IQR = pickle_data(READ, 'RIGID_3D_TRANSFORM.pickle', None)['IQR_ARRAY_LSM']

    print('Calibration candidates')
    for blob_id, points_3d in enumerate(RIGID_3D_TRANSFORM_IQR):
        print(f"{points_3d},")
        
    all_data = []
    points3D_datas = ['points3D_origin', 'points3D_legacy', 'points3D_IQR']
    colors = ['r', 'g', 'b']
    points3D_LENGTH = len(points3D_datas)
    for COMBINATION in combination_cnt:
        fig, axs = plt.subplots(3, 1, figsize=(15, 15))
        fig.suptitle(f'Calibration Data Analysis for Combination: {COMBINATION}')
        print(f"COMBINATION {COMBINATION}")
        summary_text = ""
        for idx, points3D_data in enumerate(points3D_datas):
            frame_counts, rvec_std_arr, tvec_std_arr, reproj_err_rates, label, fail_reason = STD_Analysis(points3D_data, points3D_datas, COMBINATION, CAMERA_INFO, USING_KORNIA, default_cameraK, default_dist_coeffs, camera_matrix, CAM_ID)

            axs[0].plot(frame_counts, rvec_std_arr, colors[idx]+'-', label=f'rvec std {label[idx]}', alpha=0.5)
            axs[0].plot(frame_counts, tvec_std_arr, colors[idx]+'--', label=f'tvec std {label[idx]}', alpha=0.5)
            axs[1].plot(frame_counts, reproj_err_rates, colors[idx], label=f'Reprojection error rate {label[idx]}', alpha=0.5)
            
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
            summary_text += "\n"
            all_data.append([label, COMBINATION, avg_rvec_std, std_rvec_std, avg_tvec_std, std_tvec_std, avg_reproj_err, std_reproj_err])

        axs[0].legend()
        axs[0].set_xlabel('frame_cnt')
        axs[0].set_ylabel('std')
        axs[0].set_title('Standard Deviation of rvec and tvec Magnitude per Frame')

        axs[1].legend()
        axs[1].set_xlabel('frame_cnt')
        axs[1].set_ylabel('error rate')
        axs[1].set_title('Mean Reprojection Error Rate per Frame')

        axs[2].axis('off')
        axs[2].text(0, 0, summary_text, fontsize=10)

        for ax in axs[:2]:
            ax.set_xticks(ax.get_xticks()[::5])

        plt.subplots_adjust(hspace=0.5)

    plt.show()

if __name__ == "__main__":
    # 필요한 변수 초기화
    USING_KORNIA = NOT_SET
    CAM_ID = 0  # 예시값, 실제 값으로 대체해야 함

    combination_cnt = [4, 5]  # 예시
    new_Check_Calibration_data_combination(combination_cnt, 'CAMERA_INFO.pickle', USING_KORNIA, default_cameraK, default_dist_coeffs, camera_matrix, CAM_ID)

    print('########## DONE ##########')