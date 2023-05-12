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
                                    }

    # for key, value in CAMERA_INFO.items():
    #     print(key, value)

    bboxes = []
    json_file = ''.join(['../jsons/blob_area.json'])
    json_data = rw_json_data(READ, json_file, None)
    if json_data != ERROR:
        bboxes = json_data['bboxes']

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
            METHOD = POSE_ESTIMATION_METHOD.SOLVE_PNP_AP3P

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

            # 오른쪽 그리드를 2x2로 분할
            gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1])

            # 분할된 오른쪽 그리드에 subplot 할당
            ax2 = plt.subplot(gs_sub[0, 0])
            ax3 = plt.subplot(gs_sub[0, 1])
            ax4 = plt.subplot(gs_sub[1, 0])
            ax5 = plt.subplot(gs_sub[1, 1])

            if len(bboxes) > 0:
                for i, data in enumerate(bboxes):
                    (x, y, w, h) = data['bbox']
                    IDX = int(data['idx'])
                    cam_id = 0
                    cx, cy = find_center(IMG_GRAY, (x, y, w, h))
                    if cx >= CAP_PROP_FRAME_WIDTH:
                        cx -= CAP_PROP_FRAME_WIDTH
                        cam_id = 1
                    CAMERA_INFO[f'{cam_id}']['points2D']['greysum'].append([cx, cy])
                    CAMERA_INFO[f'{cam_id}']['points3D'].append(led_data[IDX])

                for cam_id, cam_data in CAMERA_INFO.items():
                    print('\n\n')
                    # cam_id에 따라 서브플롯 선택
                    if int(cam_id) == 0:
                        ax = ax2
                    else:
                        ax = ax3

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
                            projectionMatrix = np.loadtxt(data_path)
                            intrinsic, rotationMatrix, homogeneousTranslationVector = cv2.decomposeProjectionMatrix(
                                projectionMatrix)[:3]
                            camT = -cv2.convertPointsFromHomogeneous(homogeneousTranslationVector.T)
                            camR = Rot.from_matrix(rotationMatrix)
                            blender_tvec = camR.apply(camT.ravel())
                            blender_rvec = camR.as_rotvec()
                            print('RT from Blender to Opencv')
                            print('rvecs\n', blender_rvec)
                            print('tvecs\n', blender_tvec)
                            cam_data['blender_rt']['rvec'] = blender_rvec.reshape(-1, 1)
                            cam_data['blender_rt']['tvec'] = blender_tvec.reshape(-1, 1)
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
                                    ax.text(x, y, f'{i}', fontsize=12, ha='right', va='bottom')

                                # 좌표 사이의 거리를 직선으로 표시
                                for a, b in zip(image_points, blender_image_points):
                                    ax.plot([a[0], b[0]], [a[1], b[1]], linestyle=':', color='green', alpha=0.6)
                                    distance = np.linalg.norm(a - b)
                                    midpoint = (a + b) / 2
                                    ax.text(midpoint[0], midpoint[1], f"{distance:.2f}", fontsize=12, ha='left',
                                            va='bottom',
                                            color='green', alpha=0.6)

                                ax.set_xlabel('X-axis')
                                ax.set_ylabel('Y-axis')
                                ax.set_title('2D-Point distance CAM ' + str(cam_id))
                                ax.legend()
                                ax.grid()
                                # Y축 방향 반전
                                ax.invert_yaxis()
                    cv2.imshow(f"{cam_id}_result", cam_data['image'])

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
                origin_pts = np.array(led_data).reshape(-1, 3)
                print('origin_pts\n', points3D.reshape(-1, 3))
                ax1.scatter(origin_pts[:, 0], origin_pts[:, 1], origin_pts[:, 2], color='black', alpha=0.5, marker='o', s=10)
                ax1.scatter(opencv_remake[:, 0], opencv_remake[:, 1], opencv_remake[:, 2], color='red', alpha=0.5, marker='o', s=10)
                ax1.scatter(blender_remake[:, 0], blender_remake[:, 1], blender_remake[:, 2], color='blue', alpha=0.5, marker='o', s=10)

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
                dist_blender = np.linalg.norm(points3D.reshape(-1, 3) - blender_remake, axis=1)

                ax4.bar(range(len(dist_opencv)), dist_opencv)
                ax4.set_title('Distance between origin_pts and opencv_remake')
                ax4.set_xlabel('Index')
                ax4.set_ylabel('Distance')

                ax5.bar(range(len(dist_blender)), dist_blender)
                ax5.set_title('Distance between origin_pts and blender_remake')
                ax5.set_xlabel('Index')
                ax5.set_ylabel('Distance')

                plt.tight_layout()

                plt.show()

        cv2.imshow('Image', STACK_FRAME)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    os_name = platform.system()
    image_path = '../blender_3d/image_output'
    images, image_files, data_files = load_data(image_path)
    display_tracker_research(images, image_files, data_files)
