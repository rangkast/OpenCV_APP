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


def display_tracker(images, image_files, data_files):
    index = 0
    print('data_files')
    for data_file in data_files:
        print(data_file)
    print('image_files')
    for image_file in image_files:
        print(image_file)

    # 1st Step
    # make blob box area

    bboxes = []
    while True:
        # 파일 이름을 이미지 상단에 표시
        img_name = os.path.basename(image_files[index])
        # print('image_name', img_name)
        parsed_array = img_name.split('.png')
        name_key = parsed_array[0]

        json_file = ''.join(['../jsons/' f'{name_key}.json'])
        json_data = rw_json_data(READ, json_file, None)
        if json_data != ERROR:
            bboxes = json_data['bboxes']

        ret, img_contour_binary = cv2.threshold(images[index], CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
        img_gray = cv2.cvtColor(img_contour_binary, cv2.COLOR_BGR2GRAY)
        img_draw = images[index].copy()
        cv2.putText(img_draw, img_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        key = cv2.waitKey(1)

        if key == ord('n'):
            bboxes.clear()
            index += 1
            if index >= len(images):
                print('end of files')
                break
        elif key == ord('a'):
            cv2.imshow('Image', img_gray)
            bbox = cv2.selectROI('Image', img_gray)
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

        if len(bboxes) > 0:
            for i, data in enumerate(bboxes):
                (x, y, w, h) = data['bbox']
                IDX = data['idx']
                cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1)
                view_camera_infos(img_draw, ''.join([f'{IDX}']), int(x), int(y) - 10)
                view_camera_infos(img_draw, ''.join(['[', f'{IDX}', '] '
                                                        , f' {x}'
                                                        , f' {y}']), 30, 70 + i * 30)

        cv2.namedWindow("Image")
        cv2.imshow("Image", img_draw)
    cv2.destroyAllWindows()

    # 2nd step
    # capture and calculate data

    print('Selected bounding boxes {}'.format(bboxes))
    # Specify the tracker type
    trackerType = "CSRT"
    # Create MultiTracker object
    multiTracker = cv2.legacy.MultiTracker_create()

    tracker_start = 0

    while True:
        # 파일 이름을 이미지 상단에 표시
        if index >= len(images):
            print('end of files')
            break
        img_name = os.path.basename(image_files[index])

        ret, img_contour_binary = cv2.threshold(images[index], CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
        img_gray = cv2.cvtColor(img_contour_binary, cv2.COLOR_BGR2GRAY)
        img_draw = images[index].copy()
        cv2.putText(img_draw, img_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # Initialize MultiTracker
        if tracker_start == 0:
            for i, data in enumerate(bboxes):
                multiTracker.add(createTrackerByName(trackerType), img_gray, data['bbox'])

        tracker_start = 1

        # get updated location of objects in subsequent frames
        qq, boxes = multiTracker.update(img_gray)

        # draw tracked objects
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(img_draw, p1, p2, 255, 1)
            IDX = bboxes[i]['idx']
            view_camera_infos(img_draw, ''.join([f'{IDX}']), int(newbox[0]), int(newbox[1]) - 10)

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
            if len(bboxes) > 0:
                image_points = []
                object_points = []
                except_pos = 0
                for i, data in enumerate(bboxes):
                    (x, y, w, h) = data['bbox']
                    IDX = int(data['idx'])
                    except_pos += 1
                    if except_pos == len(bboxes) / 2:
                        color = (255, 255, 255)
                        line_width = 1
                        except_pos = 0
                    else:
                        color = (0, 255, 0)
                        line_width = 2

                    cx, cy = find_center(img_gray, (x, y, w, h))
                    image_points.append([cx, cy])
                    object_points.append(led_data[IDX])

                points2D = np.array(image_points, dtype=np.float64)
                points3D = np.array(object_points, dtype=np.float64)
                print('point_3d\n', points3D)
                print('point_2d\n', points2D)

                greysum_points = np.squeeze(points2D)
                plt.scatter(greysum_points[:, 0], greysum_points[:, 1], c='black', alpha=0.5, label='GreySum')
                for g_point in greysum_points:
                    cv2.circle(img_draw, (int(g_point[0]), int(g_point[1])), 1, (0, 0, 0), -1)

                parsed_array = img_name.split('_')
                cam_id = int(parsed_array[1])
                camera_k = camera_matrix[cam_id][0]

                # blender에서 왜곡계수 처리방법 확인중
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
                print('rvecs\n', rvec.ravel())
                print('tvecs\n', tvec.ravel())

                # 3D 점들을 2D 이미지 평면에 투영
                image_points, _ = cv2.projectPoints(points3D, rvec, tvec, camera_k, dist_coeff)
                image_points = np.squeeze(image_points)
                plt.scatter(image_points[:, 0], image_points[:, 1], c='red', alpha=0.5, label='OpenCV')

                # 투영된 2D 이미지 점 출력
                print("Projected 2D image points:")
                print(image_points)

                # 빨간색점은 opencv pose-estimation 투영 결과
                for repr_blob in image_points:
                    cv2.circle(img_draw, (int(repr_blob[0]), int(repr_blob[1])), 1, (0, 0, 255), -1)

                parsed_array = img_name.split('.png')
                name_key = parsed_array[0]

                for data_path in data_files:
                    if name_key in data_path:
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
                        blender_image_points, _ = cv2.projectPoints(points3D, blender_rvec, blender_tvec, camera_k,
                                                                    dist_coeff)
                        blender_image_points = np.squeeze(blender_image_points)
                        plt.scatter(blender_image_points[:, 0], blender_image_points[:, 1], c='blue', alpha=0.5,
                                    label='Blender')

                        print("Projected 2D image points:")
                        print(blender_image_points)
                        for repr_blob in blender_image_points:
                            cv2.circle(img_draw, (int(repr_blob[0]), int(repr_blob[1])), 1, (255, 0, 0), -1)
                            # 각 점에 레이블 표시
                            for i, (x, y) in enumerate(image_points):
                                plt.text(x, y, f'{i}', fontsize=12, ha='right', va='bottom')

                            # 좌표 사이의 거리를 직선으로 표시
                            for a, b in zip(image_points, blender_image_points):
                                plt.plot([a[0], b[0]], [a[1], b[1]], linestyle=':', color='green', alpha=0.6)
                                distance = np.linalg.norm(a - b)
                                midpoint = (a + b) / 2
                                plt.text(midpoint[0], midpoint[1], f"{distance:.2f}", fontsize=12, ha='left',
                                         va='bottom',
                                         color='green', alpha=0.6)

                            plt.xlabel('X-axis')
                            plt.ylabel('Y-axis')
                            plt.title('Plot of Coordinate Groups A and B')
                            plt.legend()
                            plt.grid()
                            # Y축 방향 반전
                            plt.gca().invert_yaxis()
                cv2.imshow('Result', img_draw)
                plt.show()

        cv2.imshow('Image', img_draw)

    cv2.destroyAllWindows()
    bboxes.clear()


if __name__ == "__main__":
    os_name = platform.system()
    image_path = '../blender_3d/render_output'
    images, image_files, data_files = load_data(image_path)
    display_tracker(images, image_files, data_files)
