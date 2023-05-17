from function import *
from definition import *
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
CV_MIN_THRESHOLD = 100
CV_MAX_THRESHOLD = 255

if __name__ == "__main__":
    # CAM_INFO = pickle_data(READ, './data_result.pickle', None)
    os_name = platform.system()
    image_path = '../blender_3d/image_output'
    image_files = glob.glob(os.path.join(image_path, "*.png"))
    data_files = glob.glob(os.path.join(image_path, "*.json"))
    txt_files = glob.glob(os.path.join(image_path, "*.txt"))
    images = [cv2.imread(img) for img in image_files]

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
                                    'points2D': {'greysum': [], 'opencv': [], 'blender': []},
                                    'points3D': [],
                                    'opencv_rt': {'rvec': [], 'tvec': []},
                                    'blender_rt': {'rvec': [], 'tvec': []},
                                    'test_rt': {'rvec': [], 'tvec': []},
                                    }

    root = tk.Tk()
    width_px = root.winfo_screenwidth()
    height_px = root.winfo_screenheight()

    # 모니터 해상도에 맞게 조절
    mpl.rcParams['figure.dpi'] = 120  # DPI 설정
    monitor_width_inches = width_px / mpl.rcParams['figure.dpi']  # 모니터 너비를 인치 단위로 변환
    monitor_height_inches = height_px / mpl.rcParams['figure.dpi']  # 모니터 높이를 인치 단위로 변환

    fig = plt.figure(figsize=(monitor_width_inches, monitor_height_inches), num='Compare Center')
    # Single Axes 생성
    ax = fig.add_subplot(1, 1, 1)  # 1x1 그리드에서 첫 번째 위치에 Axes 생성

    bboxes = []

    json_file = ''.join(['../jsons/blob_area.json'])
    json_data = rw_json_data(READ, json_file, None)
    if json_data != ERROR:
        bboxes = json_data['bboxes']

    # DRAW_IMG_0 = images[1]
    # DRAW_IMG_1 = images[0]
    DRAW_IMG_0 = CAMERA_INFO['0']['image']
    DRAW_IMG_1 = CAMERA_INFO['1']['image']

    STACK_FRAME = np.hstack((DRAW_IMG_0, DRAW_IMG_1))
    cv2.putText(STACK_FRAME, CAMERA_INFO['0']['img_name'], (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(STACK_FRAME, CAMERA_INFO['1']['img_name'], (10 + CAP_PROP_FRAME_WIDTH, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    ret, img_filtered = cv2.threshold(STACK_FRAME, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
    IMG_GRAY = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)

    results = [[] for _ in range(4)]

    if len(bboxes) > 0:
        for data in bboxes:
            (x, y, w, h) = data['bbox']

            # 무게 중심 계산
            roi = IMG_GRAY[y:y + h, x:x + w]
            M = cv2.moments(roi)
            cX = int(M["m10"] / M["m00"]) + x
            cY = int(M["m01"] / M["m00"]) + y
            results[0].append((cX, cY))

            # GreySum
            gcx, gcy = find_center(IMG_GRAY, (x, y, w, h))
            results[1].append((gcx, gcy))

            # 컨투어 찾기
            contours, _ = cv2.findContours(roi.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if len(cnt) > 5:
                    ellipse = cv2.fitEllipse(cnt)
                    ellipse = ((ellipse[0][0] + x, ellipse[0][1] + y), ellipse[1], ellipse[2])  # 타원의 중심 위치를 보정합니다.
                    ax.add_patch(patches.Ellipse(ellipse[0], ellipse[1][0], ellipse[1][1], angle=ellipse[2], fill=False,
                                                 edgecolor='g'))

                    # 타원의 장축과 단축 길이, 회전 각도 가져오기
                    center, (major_axis, minor_axis), angle = ellipse

                    # 타원의 장축의 양 끝점 계산
                    dx_major = major_axis / 2 * np.cos(np.deg2rad(angle))
                    dy_major = major_axis / 2 * np.sin(np.deg2rad(angle))
                    x_major = [center[0] - dx_major, center[0] + dx_major]
                    y_major = [center[1] - dy_major, center[1] + dy_major]

                    # 타원의 단축의 양 끝점 계산
                    dx_minor = minor_axis / 2 * np.sin(np.deg2rad(angle))
                    dy_minor = minor_axis / 2 * np.cos(np.deg2rad(angle))
                    x_minor = [center[0] + dx_minor, center[0] - dx_minor]
                    y_minor = [center[1] - dy_minor, center[1] + dy_minor]

                    # 타원의 장축과 단축 그리기
                    ax.plot(x_major, y_major, color='r', alpha=0.5, linewidth=1)
                    ax.plot(x_minor, y_minor, color='b', alpha=0.5, linewidth=1)

        # 결과 출력
        for cxy in results:
            print(f"Centroid of the light source is at: {cxy}")

    ax.imshow(IMG_GRAY, cmap='gray')

    print('len(result)', len(results[0]))
    for i in range(len(results[0])):
        # 주어진 영역을 잘라내기
        (x, y, w, h) = bboxes[i]['bbox']
        cam_id = int(bboxes[i]['id'])
        # ADD other algo here
        ax.scatter(results[0][i][0], results[0][i][1], color='red', marker='o', s=3)
        ax.scatter(results[1][i][0], results[1][i][1], color='black', marker='o', s=3)

        # Draw the bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # ax.scatter(np.array(CAM_INFO['CAMERA_INFO_0']['points2D']['greysum'])[:, 0],
    #            np.array(CAM_INFO['CAMERA_INFO_0']['points2D']['greysum'])[:, 1], c='black', alpha=0.5, label='GreySum',
    #            s=1)
    # ax.scatter(np.array(CAM_INFO['CAMERA_INFO_1']['points2D']['greysum'])[:, 0] + CAP_PROP_FRAME_WIDTH,
    #            np.array(CAM_INFO['CAMERA_INFO_1']['points2D']['greysum'])[:, 1], c='black', alpha=0.5, label='GreySum',
    #            s=1)
    #
    # ax.scatter(np.array(CAM_INFO['CAMERA_INFO_0']['points2D']['opencv'])[:, 0],
    #            np.array(CAM_INFO['CAMERA_INFO_0']['points2D']['opencv'])[:, 1], c='red', alpha=0.5, label='OpenCV', s=1)
    # ax.scatter(np.array(CAM_INFO['CAMERA_INFO_1']['points2D']['opencv'])[:, 0] + CAP_PROP_FRAME_WIDTH,
    #            np.array(CAM_INFO['CAMERA_INFO_1']['points2D']['opencv'])[:, 1], c='red', alpha=0.5, label='OpenCV', s=1)
    #
    # ax.scatter(np.array(CAM_INFO['CAMERA_INFO_0']['points2D']['blender'])[:, 0],
    #            np.array(CAM_INFO['CAMERA_INFO_0']['points2D']['blender'])[:, 1], c='blue', alpha=0.5, label='Blender',
    #            s=1)
    # ax.scatter(np.array(CAM_INFO['CAMERA_INFO_1']['points2D']['blender'])[:, 0] + CAP_PROP_FRAME_WIDTH,
    #            np.array(CAM_INFO['CAMERA_INFO_1']['points2D']['blender'])[:, 1], c='blue', alpha=0.5, label='Blender',
    #            s=1)

    points3D = np.array([0.05, 0, 0])
    points3D = np.array(points3D, dtype=np.float64)
    for cam_id, cam_data in CAMERA_INFO.items():
        camera_k = cam_data['camera_k']
        dist_coeff = default_dist_coeffs
        for data_path in data_files:
            if cam_data['img_name'] in data_path:

                for txt_data in txt_files:
                    if cam_data['img_name'] in txt_data:
                        # TEXT Filder Loading
                        projectionMatrix = np.loadtxt(txt_data)
                        intrinsic, rotationMatrix, homogeneousTranslationVector = cv2.decomposeProjectionMatrix(
                            projectionMatrix)[:3]
                        camT = -cv2.convertPointsFromHomogeneous(homogeneousTranslationVector.T)
                        camR = Rot.from_matrix(rotationMatrix)
                        blender_tvec = camR.apply(camT.ravel())
                        blender_rvec = camR.as_rotvec()
                        blender_rvec = blender_rvec.reshape(-1, 1)
                        blender_tvec = blender_tvec.reshape(-1, 1)

                        blender_image_points, _ = cv2.projectPoints(points3D, blender_rvec, blender_tvec,
                                                                    cameraMatrix=intrinsic,
                                                                    distCoeffs=None)
                        blender_image_points = blender_image_points.reshape(-1, 2)
                        if int(cam_id) == 1:
                            bx = blender_image_points[:, 0] + CAP_PROP_FRAME_WIDTH
                        else:
                            bx = blender_image_points[:, 0]
                        by = blender_image_points[:, 1]

                        print('bx by', bx, by)
                        ax.scatter(bx, by, c='magenta', alpha=1, label='Blender', s=5)

                        print("Projected 2D image points <Projection>")
                        print(blender_image_points)

                json_data = rw_json_data(READ, data_path, None)

                blender_rvec = np.array(json_data['rvec']).reshape(-1, 1)
                blender_tvec = np.array(json_data['tvec']).reshape(-1, 1)

                print('RT from Blender to Opencv')
                print('rvecs\n', blender_rvec.ravel())
                print('tvecs\n', blender_tvec.ravel())
                cam_data['blender_rt']['rvec'] = blender_rvec
                cam_data['blender_rt']['tvec'] = blender_tvec

                blender_image_points, _ = cv2.projectPoints(points3D, blender_rvec, blender_tvec,
                                                            camera_k,
                                                            dist_coeff)
                blender_image_points = blender_image_points.reshape(-1, 2)
                if int(cam_id) == 1:
                    bx = blender_image_points[:, 0] + CAP_PROP_FRAME_WIDTH
                else:
                    bx = blender_image_points[:, 0]
                by = blender_image_points[:, 1]

                print('bx by', bx, by)
                # ax.scatter(bx, by, c='blue', alpha=1, label='Blender', s=5)

                print("Projected 2D image points <BLENDER RT only>")
                print(blender_image_points)

    # if len(bboxes) > 0:
    #     except_pos = 0
    #     for i, data in enumerate(bboxes):
    #         (x, y, w, h) = data['bbox']
    #         IDX = data['idx']
    #         except_pos += 1
    #         if except_pos == len(bboxes) / 2:
    #             color = (255, 255, 255)
    #             line_width = 1
    #             except_pos = 0
    #         else:
    #             color = (0, 255, 0)
    #             line_width = 2
    #
    #         cv2.rectangle(IMG_GRAY, (int(x), int(y)), (int(x + w), int(y + h)), color, line_width, 1)

    # cv2.imshow('image', IMG_GRAY)

    plt.show()
