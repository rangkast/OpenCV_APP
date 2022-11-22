import cv2

from definition import *


def view_camera_infos(frame, text, x, y):
    cv2.putText(frame, text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), lineType=cv2.LINE_AA)


def camera_setting():
    print('start open_camera')

    cap1 = cv2.VideoCapture(CAM_1[0])
    cap1_name = CAM_1[1]
    cap2 = cv2.VideoCapture(CAM_2[0])
    cap2_name = CAM_2[1]

    width = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('cap1 size: %d, %d' % (width, height))
    data_info_dictionary['display']['left'][0] = width
    data_info_dictionary['display']['left'][1] = height

    width = cap2.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('cap2 size: %d, %d' % (width, height))
    data_info_dictionary['display']['right'][0] = width
    data_info_dictionary['display']['right'][1] = height

    if not cap1.isOpened() or not cap2.isOpened():
        sys.exit()

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break
        imgL = Rotate(frame1, 270)
        imgR = Rotate(frame2, 270)
        view_camera_infos(imgL, f'{cap1_name}', 30, 35)
        cv2.circle(imgL, (int(CAP_PROP_FRAME_WIDTH / 2), int(CAP_PROP_FRAME_HEIGHT / 2)), 2, color=(0, 0, 255),
                   thickness=-1)
        cv2.imshow('left camera', imgL)
        view_camera_infos(imgR, f'{cap2_name}', 30, 35)
        cv2.circle(imgR, (int(CAP_PROP_FRAME_WIDTH / 2), int(CAP_PROP_FRAME_HEIGHT / 2)), 2, color=(0, 0, 255),
                   thickness=-1)
        cv2.imshow("right camera", imgR)

        # alpha = 0.5
        #
        # after_frame = cv2.addWeighted(frame1, alpha, frame2, alpha, 0)
        # cv2.imshow('stereo camera', after_frame)

        cv2.waitKey(CAM_DELAY)

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


def calibrate_camera():
    print('start open_camera')

    capture_cnt = 0
    # Defining the world coordinates for 3D points
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    img_ptsL = []
    img_ptsR = []
    obj_pts = []

    cap1 = cv2.VideoCapture(CAM_1[0])
    cap2 = cv2.VideoCapture(CAM_2[0])

    if not cap1.isOpened() or not cap2.isOpened():
        sys.exit()

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        imgL = Rotate(frame1, 270)
        imgR = Rotate(frame2, 270)

        if cv2.waitKey(1) == ord('c'):
            imgL_gray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            imgR_gray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
            ret, c_l, c_r = stereo_camera_calbiration(imgL, imgR, imgL_gray, imgR_gray)
            if ret == DONE:
                print('captured')
                capture_cnt += 1
                obj_pts.append(objp)
                img_ptsL.append(c_l)
                img_ptsR.append(c_r)

        elif cv2.waitKey(1) == ord('e'):
            print('start calibrate')
            if USE_EXTERNAL_TOOL_CALIBRAION == DISABLE:
                # Calibrating left camera
                retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(obj_pts, img_ptsL,
                                                                        (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT),
                                                                        None,
                                                                        None)
                hL, wL = imgL_gray.shape[:2]
                print('hL ', hL, ' wL', wL)
                new_mtxL, roiL = cv2.getOptimalNewCameraMatrix(mtxL, distL,
                                                               (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT),
                                                               1, (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT))

                # Calibrating right camera
                retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(obj_pts, img_ptsR,
                                                                        (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT),
                                                                        None,
                                                                        None)
                hR, wR = imgR_gray.shape[:2]
                new_mtxR, roiR = cv2.getOptimalNewCameraMatrix(mtxR, distR,
                                                               (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT),
                                                               1, (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT))

                view_camera_infos(imgL, f'{CAM_1[1]}' + ' calibrated', 30, 35)
                view_camera_infos(imgR, f'{CAM_2[1]}' + ' calibrated', 30, 35)

                print('new_mtxL ', new_mtxL)
                print('distL ', distL)

                print('new_mtxR ', new_mtxR)
                print('distR ', distR)
            else:
                # load json file
                json_file = ''.join(['jsons/test_result/', f'{EXTERNAL_TOOL_CALIBRATION}'])
                external_jdata = rw_json_data(READ, json_file, None)
                new_mtxL = np.array(external_jdata['stereol']['camera_k'], dtype=np.float64)
                distL = np.array(external_jdata['stereol']['distcoeff'], dtype=np.float64)
                new_mtxR = np.array(external_jdata['stereol']['camera_k'], dtype=np.float64)
                distR = np.array(external_jdata['stereol']['distcoeff'], dtype=np.float64)

            # TEST camera calibration
            flags = 0
            flags |= cv2.CALIB_FIX_INTRINSIC
            ret, CM1, dist1, CM2, dist2, R, T, E, F = cv2.stereoCalibrate(obj_pts, img_ptsL, img_ptsR,
                                                                          new_mtxL, distL,
                                                                          new_mtxR, distR,
                                                                          (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT),
                                                                          CRITERIA,
                                                                          flags)

            print(CM1, ' ', dist1)
            print(CM2, ' ', dist2)
            print('R ', R, '\nT ', T, '\nE ', E, '\nF ', F)

            rectify_scale = 1
            rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv2.stereoRectify(CM1, dist1, CM2, dist2,
                                                                                      (CAP_PROP_FRAME_WIDTH,
                                                                                       CAP_PROP_FRAME_HEIGHT),
                                                                                      R, T,
                                                                                      rectify_scale, (0, 0))

            print('rect_l ', rect_l)
            print('rect_r ', rect_r)
            print('proj_mat_l ', proj_mat_l)
            print('proj_mat_r ', proj_mat_r)

            Left_Stereo_Map = cv2.initUndistortRectifyMap(CM1, dist1, rect_l, proj_mat_l,
                                                          (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT), cv2.CV_16SC2)
            Right_Stereo_Map = cv2.initUndistortRectifyMap(CM2, dist2, rect_r, proj_mat_r,
                                                           (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT), cv2.CV_16SC2)

            print('save rectification map')
            rw_file_storage(WRITE, RECTIFY_MAP, Left_Stereo_Map, Right_Stereo_Map)

            print('save stereo info')
            json_file = ''.join(['jsons/test_result/', f'{JSON_FILE}'])
            group_data = OrderedDict()
            group_data['stereoRectify'] = {
                'rect_l': rect_l.tolist(),
                'rect_r': rect_r.tolist(),
                'proj_mat_l': proj_mat_l.tolist(),
                'proj_mat_r': proj_mat_r.tolist(),
                'Q': Q.tolist(),
                'R': R.tolist(),
                'T': T.tolist(),
                'E': E.tolist(),
                'F': F.tolist()
            }
            group_data['stereol'] = {'cam_id': CAM_1[1],
                                     'camera_k': new_mtxL.tolist(),
                                     'distcoeff': distL.tolist()}
            group_data['stereor'] = {'cam_id': CAM_2[1],
                                     'camera_k': new_mtxR.tolist(),
                                     'distcoeff': distR.tolist()}
            rw_json_data(WRITE, json_file, group_data)

            print('done')

        elif cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break

        view_camera_infos(imgL, f'{CAM_1[1]}', 30, 35)
        view_camera_infos(imgL, f'{capture_cnt}', 30, 70)
        cv2.circle(imgL, (int(CAP_PROP_FRAME_WIDTH / 2), int(CAP_PROP_FRAME_HEIGHT / 2)), 2, color=(0, 0, 255),
                   thickness=-1)
        cv2.imshow('left camera', imgL)

        view_camera_infos(imgR, f'{CAM_2[1]}', 30, 35)
        view_camera_infos(imgR, f'{capture_cnt}', 30, 70)
        cv2.circle(imgR, (int(CAP_PROP_FRAME_WIDTH / 2), int(CAP_PROP_FRAME_HEIGHT / 2)), 2, color=(0, 0, 255),
                   thickness=-1)
        cv2.imshow("right camera", imgR)
        # alpha = 0.5
        #
        # after_frame = cv2.addWeighted(frame1, alpha, frame2, alpha, 0)
        # cv2.imshow('stereo camera', after_frame)

        cv2.waitKey(CAM_DELAY)

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


def stereo_camera_calbiration(imgL, imgR, imgL_gray, imgR_gray):
    retR, cornersR = cv2.findChessboardCorners(imgR, CHECKERBOARD, None)
    retL, cornersL = cv2.findChessboardCorners(imgL, CHECKERBOARD, None)

    if retR and retL:
        cv2.cornerSubPix(imgR_gray, cornersR, (11, 11), (-1, -1), CRITERIA)
        cv2.cornerSubPix(imgL_gray, cornersL, (11, 11), (-1, -1), CRITERIA)

        cv2.drawChessboardCorners(imgR, CHECKERBOARD, cornersR, retR)
        cv2.drawChessboardCorners(imgL, CHECKERBOARD, cornersL, retL)

        return DONE, cornersL, cornersR
    else:
        print('not detected')
        return NOT_SET, NOT_SET, NOT_SET


def disparity_map(imgL, imgR):
    window_size = 3

    left_matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=160,
                                         blockSize=25,
                                         P1=8 * 3 * window_size ** 2,
                                         P2=32 * 3 * window_size ** 2,
                                         disp12MaxDiff=1,
                                         uniquenessRatio=15,
                                         speckleWindowSize=0,
                                         speckleRange=2,
                                         preFilterCap=63,
                                         mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                         )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    lmbda = 80000
    sigma = 1.2
    visual_multiplyer = 1.0

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    disp_left = left_matcher.compute(imgL, imgR)
    disp_right = right_matcher.compute(imgR, imgL)
    disp_left = np.int16(disp_left)
    disp_right = np.int16(disp_right)

    disp = wls_filter.filter(disp_left, imgL, None, disp_right)
    disp = cv2.normalize(src=disp, dst=disp,
                         beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    disp = np.uint8(disp)

    return disp


ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def make_point_3d(jdata, disp, imgL, imgR):
    Q = np.float32([jdata['stereoRectify']['Q'][0],
                    jdata['stereoRectify']['Q'][1],
                    jdata['stereoRectify']['Q'][2],
                    jdata['stereoRectify']['Q'][3]])

    print('Q ', Q)

    points = cv2.reprojectImageTo3D(disp, Q)

    reflect_matrix = np.identity(3)
    reflect_matrix[0] *= -1
    points = np.matmul(points, reflect_matrix)

    # extract colors from image
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)

    # filter by min disparity
    print('disp.min() ', disp.min())
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]

    # filter by dimension
    idx = np.fabs(out_points[:, 0]) < 20.0
    out_points = out_points[idx]
    out_colors = out_colors.reshape(-1, 3)
    out_colors = out_colors[idx]

    reflected_pts = np.matmul(out_points, reflect_matrix)
    if USE_EXTERNAL_TOOL_CALIBRAION == DISABLE:
        camera_k = np.array(jdata['stereor']['camera_k'], dtype=np.float64)
        dist_coeff = np.array(jdata['stereor']['distcoeff'], dtype=np.float64)
        Rvec = np.array(jdata['stereoRectify']['R'], dtype=np.float64)
        Tvec = np.array(jdata['stereoRectify']['T'], dtype=np.float64)
    else:
        json_file = ''.join(['jsons/test_result/', f'{EXTERNAL_TOOL_CALIBRATION}'])
        external_jdata = rw_json_data(READ, json_file, None)
        camera_k = np.array(external_jdata['stereor']['camera_k'], dtype=np.float64)
        dist_coeff = np.array(external_jdata['stereor']['distcoeff'], dtype=np.float64)
        Rvec = np.identity(3)
        Tvec = np.array([0., 0., 0.])

    projected_img, _ = cv2.projectPoints(reflected_pts, np.identity(3), np.array([0., 0., 0.]),
                                         camera_k, dist_coeff)

    # ToDo
    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')

    projected_img = projected_img.reshape(-1, 2)

    blank_img = np.zeros(imgL.shape, 'uint8')
    img_colors = imgR[mask][idx].reshape(-1, 3)

    # ToDo 왜 거꾸로 나오지?
    for i, pt in enumerate(projected_img):
        # Rotate?
        pt_x = int(pt[0])
        pt_y = int(pt[1])
        if pt_x > 0 and pt_y > 0:
            # use the BGR format to match the original image type
            col = (int(img_colors[i, 2]), int(img_colors[i, 1]), int(img_colors[i, 0]))
            cv2.circle(blank_img, (pt_x, pt_y), 1, col)

    return blank_img


def rectification(imgL, imgR, l_map, r_map):
    img_left = imgL.copy()
    img_right = imgR.copy()
    Left_nice = cv2.remap(img_left, l_map[0], l_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(img_right, r_map[0], r_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    return Left_nice, Right_nice


def stereo_camera_start():
    print('start open_camera')
    ret, l_map, r_map = rw_file_storage(READ, RECTIFY_MAP, NOT_SET, NOT_SET)
    # load json file
    json_file = ''.join(['jsons/test_result/', f'{JSON_FILE}'])
    jdata = rw_json_data(READ, json_file, None)

    if ret == DONE:
        cap1 = cv2.VideoCapture(CAM_1[0])
        cap2 = cv2.VideoCapture(CAM_2[0])

        if not cap1.isOpened() or not cap2.isOpened():
            sys.exit()

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break
            imgL = Rotate(frame1, 270)
            imgR = Rotate(frame2, 270)

            KEY = cv2.waitKey(1)

            # show origin frame
            # alpha = 0.5
            # origin_frame = cv2.addWeighted(imgL, alpha, imgR, alpha, 0)
            # cv2.imshow('origin frame', origin_frame)

            # Rectification
            R_L_F, R_R_F = rectification(imgL, imgR, l_map, r_map)
            out = R_R_F.copy()
            out[:, :, 0] = R_R_F[:, :, 0]
            out[:, :, 1] = R_R_F[:, :, 1]
            out[:, :, 2] = R_L_F[:, :, 2]
            cv2.imshow("rectification", out)

            # Disparity
            disparity_frame = disparity_map(R_L_F, R_R_F)
            cv2.imshow("disparity map", disparity_frame)

            if KEY == ord('c'):
                # make 3d point cloud
                point_cloud_frame = make_point_3d(jdata, disparity_frame, imgL, imgR)
                cv2.imshow("point cloud", point_cloud_frame)
            elif KEY & 0xFF == 27:  # Esc pressed
                break

            cv2.waitKey(CAM_DELAY)

        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

        # Draw point cloud
        try:
            cloud = o3d.io.read_point_cloud("out.ply")  # Read the point cloud
            o3d.visualization.draw_geometries([cloud])
        except:
            print('error occured')
