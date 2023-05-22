import numpy as np

from definition import *
from function import *
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

CAP_PROP_FRAME_WIDTH = 1280
CAP_PROP_FRAME_HEIGHT = 960
CV_MIN_THRESHOLD = 100
CV_MAX_THRESHOLD = 255


def detect_led_lights(image, padding=5):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blob_info = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # 주변 부분을 포함하기 위해 패딩을 적용
        x -= padding
        y -= padding
        w += padding * 2
        h += padding * 2
        blob_info.append([x, y, w, h])

    return blob_info


# Euclidean distance function
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


# Euclidean distance function
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))


if __name__ == "__main__":
    os_name = platform.system()
    # image_path = '../blender_3d/image_output/blender_basic/'
    # image_path = '../blender_3d/image_output/blender_test_image/'
    image_path = '../blender_3d/image_output/cylinder_base/'
    result_data = rw_json_data(READ, image_path + 'blender_test_image.json', None)
    image_files = glob.glob(os.path.join(image_path, "*blender_test_image.png"))
    data_files = glob.glob(os.path.join(image_path, "*.json"))

    images = [cv2.imread(img) for img in image_files]
    print('image_files', image_files)
    root = tk.Tk()
    width_px = root.winfo_screenwidth()
    height_px = root.winfo_screenheight()

    # 모니터 해상도에 맞게 조절
    mpl.rcParams['figure.dpi'] = 120  # DPI 설정
    monitor_width_inches = width_px / mpl.rcParams['figure.dpi']  # 모니터 너비를 인치 단위로 변환
    monitor_height_inches = height_px / mpl.rcParams['figure.dpi']  # 모니터 높이를 인치 단위로 변환

    # Create a GridSpec object
    gs = gridspec.GridSpec(2, 2)  # Change the number of rows and columns as needed

    fig = plt.figure(figsize=(monitor_width_inches, monitor_height_inches), num='Compare Center')

    # Create subplots using GridSpec
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    dist0 = fig.add_subplot(gs[1, 0])
    dist1 = fig.add_subplot(gs[1, 1])

    for cam_id, IMG in enumerate(images):
        ret, img_filtered = cv2.threshold(IMG, CV_MIN_THRESHOLD, CV_MAX_THRESHOLD, cv2.THRESH_TOZERO)
        IMG_GRAY = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2GRAY)
        results = [[] for _ in range(4)]
        blob_area = detect_led_lights(IMG_GRAY, 5)

        if cam_id == 0:
            key = 'CAMERA_0'
            ax = ax0  # Use the first subplot for cam_id 0
            dist_ax = dist0  # Use the first distance subplot for cam_id 0
        else:
            key = 'CAMERA_1'
            ax = ax1  # Use the second subplot for cam_id 1
            dist_ax = dist1  # Use the second distance subplot for cam_id 1

        for idx, area in enumerate(blob_area):
            (x, y, w, h) = area

            # 무게 중심 계산
            roi = IMG_GRAY[y:y + h, x:x + w]
            M = cv2.moments(roi)
            cX = float(M["m10"] / M["m00"]) + x
            cY = float(M["m01"] / M["m00"]) + y
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

        ax.set_title(f'2D Image and Projection of GreySum Blender Obj Util CAM_{cam_id}')
        ax.imshow(IMG_GRAY, cmap='gray')
        mapping_indices = []
        for i in range(len(results[1])):
            (x, y, w, h) = blob_area[i]
            ax.scatter(results[1][i][0], results[1][i][1], color='black', marker='o', alpha=0.7, s=5)

            # Filter the points that are within the current blob area
            within_blob_area = [j for j, coord in enumerate(result_data[key]['Obj_Util'])
                                if x <= coord[0] <= x + w and y <= coord[1] <= y + h]

            # Store the indices of the points within the blob area
            mapping_indices.append(within_blob_area)

            # Draw the bbox
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        print('mapping_indices', mapping_indices)
        total_shift_x = 0
        total_shift_y = 0
        for i in range(len(results[1])):
            for j in mapping_indices[i]:
                shift_x = result_data[key]['Obj_Util'][j][0] - results[1][i][0]
                shift_y = result_data[key]['Obj_Util'][j][1] - results[1][i][1]
                total_shift_x += shift_x
                total_shift_y += shift_y
        average_shift_x = total_shift_x / len(results[1])
        average_shift_y = total_shift_y / len(results[1])

        print('average_shift_x', average_shift_x, 'average_shift_y', average_shift_y)
        for i in np.array(mapping_indices).squeeze():
            # OpenCV ProjectionPoints
            ax.scatter(result_data[key]['OpenCV'][i][0], result_data[key]['OpenCV'][i][1], color='blue', marker='o', alpha=0.7, s=5)
            # Blender Object Util (WorldToView)
            ax.scatter(result_data[key]['Obj_Util'][i][0], result_data[key]['Obj_Util'][i][1], color='red', marker='o', alpha=0.7, s=5)
            ax.scatter(result_data[key]['Obj_Util'][i][0] - average_shift_x,
                       result_data[key]['Obj_Util'][i][1] - average_shift_y,
                       color='green',
                       marker='o',
                       alpha=0.7, s=5)
            ax.text(result_data[key]['Obj_Util'][i][0], result_data[key]['Obj_Util'][i][1] - 50, f'{i}', color='white',
                    fontsize=10)

        print('greysum')
        print(results[1])

        print('Blender Obj_Util')
        print(result_data[key]['Obj_Util'])

        # Initialize lists to store the distances
        distances_0 = []
        distances_1 = []

        for i in range(len(results[1])):
            for j in mapping_indices[i]:
                # Calculate the distance before and after applying the shift
                distance_0 = euclidean_distance(results[1][i], result_data[key]['Obj_Util'][j])
                distance_1 = euclidean_distance(results[1][i], [result_data[key]['Obj_Util'][j][0] - average_shift_x,
                                                                result_data[key]['Obj_Util'][j][
                                                                    1] - average_shift_y])
                distances_0.append(distance_0)
                distances_1.append(distance_1)

        # Calculate the mean distances
        mean_distance_0 = np.mean(distances_0)
        mean_distance_1 = np.mean(distances_1)

        # Generate indices for plotting
        indices = np.arange(len(distances_0))

        # Create a bar plot of the distances
        dist_ax.bar(indices, distances_0, width=0.4, align='center', alpha=0.5, label='Before Shift')
        dist_ax.bar(indices, distances_1, width=0.4, align='edge', alpha=0.5, label='After Shift')

        dist_ax.set_ylabel('Distance')
        dist_ax.set_title(f'Euclidean Distances Before and After Shift for Camera {cam_id}')

        # Display the mean distances
        dist_ax.axhline(y=mean_distance_0, color='b', linestyle='-',
                        label=f'Mean Distance Before Shift: {mean_distance_0:.2f}')
        dist_ax.axhline(y=mean_distance_1, color='orange', linestyle='-',
                        label=f'Mean Distance After Shift: {mean_distance_1:.2f}')

        dist_ax.legend()

        # Set the x ticks
        dist_ax.set_xticks(indices)

    plt.show()
