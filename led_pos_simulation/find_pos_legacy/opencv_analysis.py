from function import *
from definition import *
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
CV_MIN_THRESHOLD = 150
CV_MAX_THRESHOLD = 255

if __name__ == "__main__":
    os_name = platform.system()
    image_path = '../blender_3d/image_output'
    images, image_files, data_files = load_data(image_path)

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

    bboxes = []
    json_file = ''.join(['../jsons/blob_area.json'])
    json_data = rw_json_data(READ, json_file, None)
    if json_data != ERROR:
        bboxes = json_data['bboxes']

    DRAW_IMG_0 = CAMERA_INFO['0']['image']
    DRAW_IMG_1 = CAMERA_INFO['1']['image']

    STACK_FRAME = np.hstack((DRAW_IMG_0, DRAW_IMG_1))
    cv2.putText(STACK_FRAME, CAMERA_INFO['0']['img_name'], (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(STACK_FRAME, CAMERA_INFO['1']['img_name'], (10 + CAP_PROP_FRAME_WIDTH, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    ret, img_filtered = cv2.threshold(STACK_FRAME, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
    IMG_GRAY = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
    results = []

    if len(bboxes) > 0:
        for data in bboxes:
            (x, y, w, h) = data['bbox']

            # 무게 중심 계산
            roi = IMG_GRAY[y:y + h, x:x + w]
            M = cv2.moments(roi)
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]

            results.append((cX + x, cY + y))


        # 결과 출력
        for cX, cY in results:
            print(f"Centroid of the light source is at: {cX:.8f}, {cY:.8f}")

    fig, ax = plt.subplots(1, 1)
    ax.imshow(IMG_GRAY, cmap='gray')

    for i, (cX, cY) in enumerate(results):
        # 주어진 영역을 잘라내기
        (x, y, w, h) = bboxes[i]['bbox']

        # 원본 이미지에 중심점 그리기
        ax.scatter(cX, cY, color='red')

        # Draw the bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
