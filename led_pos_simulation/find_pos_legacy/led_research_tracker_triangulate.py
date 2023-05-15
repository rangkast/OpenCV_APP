from function import *
from definition import *

CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
CV_MIN_THRESHOLD = 150
CV_MAX_THRESHOLD = 255
pickle_file = './result_cylinder_base.pickle'
data = pickle_data(READ, pickle_file, None)
led_data = data['LED_INFO']
for i, leds in enumerate(led_data):
    print(f"{i}, {leds}")


def recover_euler_degree(CAMERA_INFO):
    print('\n\n')
    for cam_id, cam_data in CAMERA_INFO.items():
        print('cam_id', cam_id)

        position, rotation = blender_location_rotation_from_opencv(np.array(cam_data['opencv_rt']['rvec'].ravel()),
                                                                   np.array(cam_data['opencv_rt']['tvec'].ravel()))
        rotation = quaternion_to_euler_degree(rotation)
        print('OpenCV')
        print('euler(degree)', rotation[0], rotation[1], rotation[2])

        X, Y, Z = position
        print('position', X, Y, Z)

        position, rotation = blender_location_rotation_from_opencv(np.array(cam_data['blender_rt']['rvec'].ravel()),
                                                                   np.array(cam_data['blender_rt']['tvec'].ravel()))
        rotation = quaternion_to_euler_degree(rotation)
        print('BLENDER')
        print('euler(degree)', rotation[0], rotation[1], rotation[2])
        X, Y, Z = position
        print('position', X, Y, Z)


def display_tracker_research(images, image_files, data_files):
    print('data_files')
    for data_file in data_files:
        print(data_file)
    print('image_files')
    for image_file in image_files:
        print(image_file)

    # data parsing
    CAMERA_INFO = {}
    for idx, image_file in enumerate(image_files):
        img_name = os.path.basename(image_file)
        cam_id = int(img_name.split('_')[1])
        camera_k = camera_matrix[0][0]
        name_key = img_name.split('.png')[0]
        CAMERA_INFO[f'{cam_id}'] = {'image': images[idx],
                                    'img_name': name_key,
                                    'camera_k': camera_k,
                                    'points2D': {'greysum': [], 'blender': []},
                                    'points3D': [],
                                    'opencv_rt': {'rvec': [], 'tvec': []},
                                    'blender_rt': {'rvec': [], 'tvec': []},
                                    'test_rt': {'rvec': [], 'tvec': []},
                                    }

    # for key, value in CAMERA_INFO.items():
    #     print(key, value)

    bboxes = []
    json_file = ''.join(['../jsons/blob_area.json'])
    json_data = rw_json_data(READ, json_file, None)
    if json_data != ERROR:
        bboxes = json_data['bboxes']

    if 0:
        while True:
            DRAW_IMG_0 = CAMERA_INFO['0']['image']
            DRAW_IMG_1 = CAMERA_INFO['1']['image']

            STACK_FRAME = np.hstack((DRAW_IMG_0, DRAW_IMG_1))
            cv2.putText(STACK_FRAME, CAMERA_INFO['0']['img_name'], (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(STACK_FRAME, CAMERA_INFO['1']['img_name'], (10 + CAP_PROP_FRAME_WIDTH, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            ret, img_filtered = cv2.threshold(STACK_FRAME, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
            IMG_GRAY = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)

            if len(bboxes) > 0:
                except_pos = 0
                for i, data in enumerate(bboxes):
                    (x, y, w, h) = data['bbox']
                    IDX = data['idx']
                    except_pos += 1
                    if except_pos == len(bboxes) / 2:
                        color = (255, 255, 255)
                        line_width = 1
                        except_pos = 0
                    else:
                        color = (0, 255, 0)
                        line_width = 2

                    cv2.rectangle(STACK_FRAME, (int(x), int(y)), (int(x + w), int(y + h)), color, line_width,
                                  1)
                    view_camera_infos(STACK_FRAME, ''.join([f'{IDX}']), int(x), int(y) - 10)

            key = cv2.waitKey(1)

            if key == ord('a'):
                cv2.imshow('Image', IMG_GRAY)
                bbox = cv2.selectROI('Image', IMG_GRAY)
                while True:
                    inputs = input('input led number: ')
                    if inputs.isdigit():
                        input_number = int(inputs)
                        if input_number in range(0, 1000):
                            print('label number ', input_number)
                            print('bbox ', bbox)
                            bboxes.append({'idx': input_number, 'bbox': bbox})
                            break
                    elif cv2.waitKey(1) == ord('q'):
                        bboxes.clear()
                        break
            elif key == ord('c'):
                print('clear area')
                bboxes.clear()

            elif key == ord('s'):
                print('save blob area')
                json_data = OrderedDict()
                json_data['bboxes'] = bboxes
                # Write json data
                rw_json_data(WRITE, json_file, json_data)

            elif key & 0xFF == 27:
                print('ESC pressed')
                cv2.destroyAllWindows()
                return

            elif key == ord('e'):
                print('go next step')
                break

            cv2.imshow("Image", STACK_FRAME)

    print('Selected bounding boxes {}'.format(bboxes))
    # Specify the tracker type
    '''
    특정 버전 에서만 됨
    pip uninstall opencv-python
    pip uninstall opencv-contrib-python
    pip uninstall opencv-contrib-python-headless
    pip3 install opencv-contrib-python==4.5.5.62
    '''
    trackerType = "CSRT"
    # Create MultiTracker object
    multiTracker = cv2.legacy.MultiTracker_create()

    tracker_start = 0

    while True:
        DRAW_IMG_0 = CAMERA_INFO['0']['image']
        DRAW_IMG_1 = CAMERA_INFO['1']['image']

        STACK_FRAME = np.hstack((DRAW_IMG_0, DRAW_IMG_1))
        cv2.putText(STACK_FRAME, CAMERA_INFO['0']['img_name'], (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(STACK_FRAME, CAMERA_INFO['1']['img_name'], (10 + CAP_PROP_FRAME_WIDTH, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        ret, img_filtered = cv2.threshold(STACK_FRAME, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
        IMG_GRAY = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)

        # Initialize MultiTracker
        if tracker_start == 0:
            for i, data in enumerate(bboxes):
                multiTracker.add(createTrackerByName(trackerType), IMG_GRAY, data['bbox'])

        tracker_start = 1
        # get updated location of objects in subsequent frames
        qq, boxes = multiTracker.update(IMG_GRAY)

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(STACK_FRAME, p1, p2, 255, 1)
            IDX = bboxes[i]['idx']
            view_camera_infos(STACK_FRAME, ''.join([f'{IDX}']), int(newbox[0]), int(newbox[1]) - 10)

        key = cv2.waitKey(1)

        # quit on ESC button
        if key & 0xFF == 27:  # Esc pressed
            cv2.destroyAllWindows()
            return
        elif key == ord('e'):
            break
        elif key == ord('c'):
            print('capture start')
            METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_RANSAC

            root = tk.Tk()
            width_px = root.winfo_screenwidth()
            height_px = root.winfo_screenheight()

            # 모니터 해상도에 맞게 조절
            mpl.rcParams['figure.dpi'] = 120  # DPI 설정
            monitor_width_inches = width_px / mpl.rcParams['figure.dpi']  # 모니터 너비를 인치 단위로 변환
            monitor_height_inches = height_px / mpl.rcParams['figure.dpi']  # 모니터 높이를 인치 단위로 변환

            fig = plt.figure(figsize=(monitor_width_inches, monitor_height_inches), num='LED Position FinDer')

            # 2:1 비율로 큰 그리드 생성
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

            # 왼쪽 그리드에 subplot 할당
            ax1 = plt.subplot(gs[0], projection='3d')

            # 오른쪽 그리드를 위에는 2개, 아래는 3개로 분할
            gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], height_ratios=[1, 1])

            # 분할된 오른쪽 그리드의 위쪽에 subplot 할당
            gs_sub_top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_sub[0])
            ax2 = plt.subplot(gs_sub_top[0])
            ax3 = plt.subplot(gs_sub_top[1])

            # 분할된 오른쪽 그리드의 아래쪽에 subplot 할당
            gs_sub_bottom = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_sub[1])
            ax4 = plt.subplot(gs_sub_bottom[0])
            ax5 = plt.subplot(gs_sub_bottom[1])
            ax6 = plt.subplot(gs_sub_bottom[2])

            LED_NUMBER = []
            if len(bboxes) > 0:
                for i, data in enumerate(bboxes):
                    (x, y, w, h) = data['bbox']
                    IDX = int(data['idx'])
                    cam_id = 0
                    cx, cy = find_center(IMG_GRAY, (x, y, w, h))
                    if cx >= CAP_PROP_FRAME_WIDTH:
                        cx -= CAP_PROP_FRAME_WIDTH
                        cam_id = 1
                        LED_NUMBER.append(IDX)
                    CAMERA_INFO[f'{cam_id}']['points2D']['greysum'].append([cx, cy])
                    CAMERA_INFO[f'{cam_id}']['points3D'].append(led_data[IDX])

                for cam_id, cam_data in CAMERA_INFO.items():
                    print('\n\n')
                    # cam_id에 따라 서브플롯 선택
                    if int(cam_id) == 0:
                        ax = ax2
                    else:
                        ax = ax3

                    # Y축 방향 반전
                    ax.invert_yaxis()

                    print('CAM id', cam_id, 'ax', ax)

                    points2D = np.array(cam_data['points2D']['greysum'], dtype=np.float64)
                    points3D = np.array(cam_data['points3D'], dtype=np.float64)
                    print('point_3d\n', points3D)
                    print('point_2d\n', points2D)

                    greysum_points = np.squeeze(points2D)
                    for g_point in greysum_points:
                        cv2.circle(cam_data['image'], (int(g_point[0]), int(g_point[1])), 1, (0, 0, 0), -1)

                    camera_k = cam_data['camera_k']
                    dist_coeff = default_dist_coeffs
                    INPUT_ARRAY = [
                        cam_id,
                        points3D,
                        points2D,
                        camera_k,
                        dist_coeff
                    ]

                    ret, rvec, tvec, inliers = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)

                    print('RT from OpenCV SolvePnP')
                    print('rvec', rvec)
                    print('tvec', tvec)

                    print('rvecs\n', rvec.ravel())
                    print('tvecs\n', tvec.ravel())
                    cam_data['opencv_rt']['rvec'] = rvec
                    cam_data['opencv_rt']['tvec'] = tvec

                    # 3D 점들을 2D 이미지 평면에 투영
                    image_points, _ = cv2.projectPoints(points3D, rvec, tvec, camera_k, dist_coeff)
                    image_points = np.squeeze(image_points)

                    ax.scatter(greysum_points[:, 0], greysum_points[:, 1], c='black', alpha=0.5, label='GreySum')
                    ax.scatter(image_points[:, 0], image_points[:, 1], c='red', alpha=0.5, label='OpenCV')

                    # 투영된 2D 이미지 점 출력
                    print("Projected 2D image points:")
                    print(image_points)
                    # 빨간색점은 opencv pose-estimation 투영 결과
                    for repr_blob in image_points:
                        cv2.circle(cam_data['image'], (int(repr_blob[0]), int(repr_blob[1])), 1, (0, 0, 255), -1)

                    for data_path in data_files:
                        if cam_data['img_name'] in data_path:

                            # TEXT Filder Loading
                            # projectionMatrix = np.loadtxt(data_path)
                            # intrinsic, rotationMatrix, homogeneousTranslationVector = cv2.decomposeProjectionMatrix(
                            #     projectionMatrix)[:3]
                            # camT = -cv2.convertPointsFromHomogeneous(homogeneousTranslationVector.T)
                            # camR = Rot.from_matrix(rotationMatrix)
                            # blender_tvec = camR.apply(camT.ravel())
                            # blender_rvec = camR.as_rotvec()
                            # cam_data['blender_rt']['rvec'] = blender_rvec.reshape(-1, 1)
                            # cam_data['blender_rt']['tvec'] = blender_tvec.reshape(-1, 1)

                            json_data = rw_json_data(READ, data_path, None)

                            blender_rvec = np.array(json_data['rvec']).reshape(-1, 1)
                            blender_tvec = np.array(json_data['tvec']).reshape(-1, 1)
                            print('RT from Blender to Opencv')
                            print('rvecs\n', blender_rvec.ravel())
                            print('tvecs\n', blender_tvec.ravel())
                            cam_data['blender_rt']['rvec'] = blender_rvec
                            cam_data['blender_rt']['tvec'] = blender_tvec

                            blender_image_points, _ = cv2.projectPoints(points3D, blender_rvec, blender_tvec, camera_k,
                                                                        dist_coeff)
                            blender_image_points = np.squeeze(blender_image_points)
                            ax.scatter(blender_image_points[:, 0], blender_image_points[:, 1], c='blue', alpha=0.5,
                                       label='Blender')

                            print("Projected 2D image points:")
                            print(blender_image_points)
                            cam_data['points2D']['blender'] = blender_image_points
                            for repr_blob in blender_image_points:
                                cv2.circle(cam_data['image'], (int(repr_blob[0]), int(repr_blob[1])), 1, (255, 0, 0),
                                           -1)
                                # 각 점에 레이블 표시
                                for i, (x, y) in enumerate(image_points):
                                    ax.text(x, y, f'{LED_NUMBER[i]}', fontsize=12, ha='right', va='bottom')

                                # 좌표 사이의 거리를 직선으로 표시
                                for a, b in zip(image_points, blender_image_points):
                                    ax.plot([a[0], b[0]], [a[1], b[1]], linestyle=':', color='blue', alpha=0.6)
                                    distance = np.linalg.norm(a - b)
                                    midpoint = (a + b) / 2
                                    ax.text(midpoint[0] + 30, midpoint[1], f"{distance:.2f}", fontsize=12, ha='left',
                                            va='bottom',
                                            color='blue', alpha=0.6)

                                for a, b in zip(image_points, greysum_points):
                                    ax.plot([a[0], b[0]], [a[1], b[1]], linestyle=':', color='red', alpha=0.6)
                                    distance = np.linalg.norm(a - b)
                                    midpoint = (a + b) / 2
                                    ax.text(midpoint[0] + 10, midpoint[1], f"{distance:.2f}", fontsize=12, ha='left',
                                            va='bottom',
                                            color='red', alpha=0.6)

                                ax.set_xlabel('X-axis')
                                ax.set_ylabel('Y-axis')

                                ax.set_title('2D-Point distance CAM ' + str(cam_id))
                                ax.legend()
                                ax.grid()

                    # cv2.imshow(f"{cam_id}_result", cam_data['image'])

                # Try to remake 3D point and compare
                camera_k_0 = CAMERA_INFO['0']['camera_k']
                camera_k_1 = CAMERA_INFO['1']['camera_k']

                # OpenCV RT 3D point remake result
                opencv_remake = remake_3d_point(camera_k_0, camera_k_1,
                                                CAMERA_INFO['0']['opencv_rt'],
                                                CAMERA_INFO['1']['opencv_rt'],
                                                CAMERA_INFO['0']['points2D']['greysum'],
                                                CAMERA_INFO['1']['points2D']['greysum']).reshape(-1, 3)
                print('opencv_remake\n', opencv_remake)

                # Blender RT 3D point remake result
                blender_remake = remake_3d_point(camera_k_0, camera_k_1,
                                                 CAMERA_INFO['0']['blender_rt'],
                                                 CAMERA_INFO['1']['blender_rt'],
                                                 CAMERA_INFO['0']['points2D']['blender'],
                                                 CAMERA_INFO['1']['points2D']['blender']).reshape(-1, 3)
                print('blender_remake\n', blender_remake)


                ###################
                ####### TEST ######
                dist_coeff = default_dist_coeffs
                INPUT_ARRAY = [
                    0,
                    np.array(CAMERA_INFO['0']['points3D'], dtype=np.float64),
                    CAMERA_INFO['0']['points2D']['blender'],
                    camera_k_0,
                    dist_coeff
                ]

                ret, rvec_0, tvec_0, inliers = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
                CAMERA_INFO['0']['test_rt']['rvec'] = rvec_0
                CAMERA_INFO['0']['test_rt']['tvec'] = tvec_0
                print('rvec_0\n', rvec_0.ravel())
                print('tvec_0\n', tvec_0.ravel())

                INPUT_ARRAY = [
                    1,
                    np.array(CAMERA_INFO['1']['points3D'], dtype=np.float64),
                    CAMERA_INFO['1']['points2D']['blender'],
                    camera_k_1,
                    dist_coeff
                ]

                ret, rvec_1, tvec_1, inliers = SOLVE_PNP_FUNCTION[METHOD](INPUT_ARRAY)
                CAMERA_INFO['1']['test_rt']['rvec'] = rvec_1
                CAMERA_INFO['1']['test_rt']['tvec'] = tvec_1
                print('rvec_1\n', rvec_1.ravel())
                print('tvec_1\n', tvec_1.ravel())

                # Get R|T from blender position
                test_remake = remake_3d_point(camera_k_0, camera_k_1,
                                              CAMERA_INFO['0']['test_rt'],
                                              CAMERA_INFO['1']['test_rt'],
                                              CAMERA_INFO['0']['points2D']['blender'],
                                              CAMERA_INFO['1']['points2D']['blender']).reshape(-1, 3)
                print('test_remake\n', test_remake)

                origin_pts = np.array(led_data).reshape(-1, 3)
                print('origin_pts\n', points3D.reshape(-1, 3))
                ax1.scatter(origin_pts[:, 0], origin_pts[:, 1], origin_pts[:, 2], color='black', alpha=0.5, marker='o',
                            s=10)
                ax1.scatter(opencv_remake[:, 0], opencv_remake[:, 1], opencv_remake[:, 2], color='red', alpha=0.5,
                            marker='o', s=10)
                ax1.scatter(blender_remake[:, 0], blender_remake[:, 1], blender_remake[:, 2], color='blue', alpha=0.5,
                            marker='o', s=10)
                ax1.scatter(test_remake[:, 0], test_remake[:, 1], test_remake[:, 2], color='green', alpha=0.5,
                            marker='o', s=10)
                for index in range(blender_remake.shape[0]):
                    ax1.text(blender_remake[index, 0], blender_remake[index, 1], blender_remake[index, 2],
                             f'{LED_NUMBER[index]}', size=5)

                ax1.scatter(0, 0, 0, marker='o', color='k', s=20)
                ax1.set_xlim([-0.1, 0.1])
                ax1.set_xlabel('X')
                ax1.set_ylim([-0.1, 0.1])
                ax1.set_ylabel('Y')
                ax1.set_zlim([-0.1, 0.1])
                ax1.set_zlabel('Z')
                scale = 1.5
                f = zoom_factory(ax1, base_scale=scale)

                # Compute Euclidean distance
                dist_opencv = np.linalg.norm(points3D.reshape(-1, 3) - opencv_remake, axis=1)
                print('dist_opencv\n', dist_opencv)
                dist_blender = np.linalg.norm(points3D.reshape(-1, 3) - blender_remake, axis=1)
                print('dist_blender\n', dist_blender)
                dist_test = np.linalg.norm(points3D.reshape(-1, 3) - test_remake, axis=1)
                print('dist_test\n', dist_test)
                # Find max values
                max_opencv = np.max(dist_opencv)
                max_blender = np.max(dist_blender)
                ax4.bar(range(len(dist_opencv)), dist_opencv, width=0.4)
                ax4.set_title('Distance between origin_pts and opencv_remake')
                ax4.set_xlabel('LEDS')
                ax4.set_ylabel('Distance')
                ax4.set_xticks(range(len(dist_opencv)))
                ax4.set_xticklabels(LED_NUMBER)
                ax4.set_yscale('log')

                ax5.bar(range(len(dist_blender)), dist_blender, width=0.4)
                ax5.set_title('Distance between origin_pts and blender_remake')
                ax5.set_xlabel('LEDS')
                ax5.set_ylabel('Distance')
                ax5.set_xticks(range(len(dist_blender)))
                ax5.set_xticklabels(LED_NUMBER)
                ax5.set_yscale('log')

                ax6.bar(range(len(dist_test)), dist_test, width=0.4)
                ax6.set_title('Distance between origin_pts and test_remake')
                ax6.set_xlabel('LEDS')
                ax6.set_ylabel('Distance')
                ax6.set_xticks(range(len(dist_test)))
                ax6.set_xticklabels(LED_NUMBER)
                ax6.set_yscale('log')

                # plt.tight_layout()

                recover_euler_degree(CAMERA_INFO)

                plt.show()

        cv2.imshow('Image', STACK_FRAME)

    file = './data_result.pickle'
    result_data = OrderedDict()
    result_data['CAMERA_INFO_0'] = CAMERA_INFO['0']
    result_data['CAMERA_INFO_1'] = CAMERA_INFO['1']
    ret = pickle_data(WRITE, file, result_data)
    if ret != ERROR:
        print('result_data saved')

    cv2.destroyAllWindows()


if __name__ == "__main__":
    os_name = platform.system()
    image_path = '../blender_3d/image_output'
    images, image_files, data_files = load_data(image_path)
    display_tracker_research(images, image_files, data_files)
