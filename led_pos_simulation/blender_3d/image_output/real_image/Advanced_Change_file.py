from Advanced_Function import *    


DATA_SET_PATH = f"{script_dir}/../../../../../dataset"
COMPARE_DATA_SET_PATH = f"{DATA_SET_PATH}/dataset_20cm_400us"
RENDER_DATA_PATH = f"{script_dir}/tmp/render"

# 이미지 파일로 저장
def mov_to_bmp(LR):
    def extract_frames(video_path, output_path):
        # 비디오 파일 열기
        video = cv2.VideoCapture(video_path)

        # 비디오 파일이 열렸는지 확인
        if not video.isOpened():
            print("비디오 파일을 열 수 없습니다.")
            return

        # 프레임 수와 프레임 속도 가져오기
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)

        # 각 프레임을 BMP 파일로 저장
        for i in range(frame_count):
            # 프레임 읽기
            ret, frame = video.read()

            # 프레임 읽기에 실패하면 종료
            if not ret:
                break

            # BMP 파일로 저장
            frame_path = f"{output_path}/frame_{i:04}.bmp"
            cv2.imwrite(frame_path, frame)

            # 진행 상황 출력
            print(f"프레임 {i:04}/{frame_count} 저장 완료")

        # 비디오 파일 닫기
        video.release()

    # .mkv 파일 경로와 프레임을 저장할 폴더 경로 설정
    if LR == 0:
        video_path =  f"{RENDER_DATA_PATH}/0035-0155_L.mkv"
        output_path =  f"{DATA_SET_PATH}/render/left"
    else:
        video_path =  f"{RENDER_DATA_PATH}/0035-0155_R.mkv"
        output_path =  f"{DATA_SET_PATH}/render/right"

    # 프레임 추출 함수 호출
    extract_frames(video_path, output_path)



def change_file_name(LR):
    # 이름 변경
    script_dir = os.path.dirname(os.path.realpath(__file__))
    # Add the directory containing poselib to the module search path
    print(script_dir)

    read_floder = f"{COMPARE_DATA_SET_PATH}/"
    write_folder = f"{DATA_SET_PATH}/dataset/"
    if LR == 0:
        cam_image_files = sorted(glob.glob(read_floder + 'CAM0_*.bmp'))
    else:
        cam_image_files = sorted(glob.glob(read_floder + 'CAM1_*.bmp'))

    file_name = []
    for bmps in cam_image_files:
        split_file = bmps.split('/')
        file_name.append(split_file[len(split_file) - 1])

    if LR == 0:
        image_files = sorted(glob.glob(f"{DATA_SET_PATH}/render/left/*.bmp"))
    else:
        image_files = sorted(glob.glob(f"{DATA_SET_PATH}/render/right/*.bmp"))
    print(len(image_files), ' ', len(cam_image_files))

    for i, images in enumerate(image_files):
        print(f"{i}, {images}")
        frame = cv2.imread(images)
        cv2.imwrite(f"{write_folder}{file_name[i]}", frame)


def change_json_data(LR):
    # json 데이터 변경

    if LR == 0:
        json_file =  f"{COMPARE_DATA_SET_PATH}/Capture0_0.json"
        json_file_write = f"{DATA_SET_PATH}/dataset/Capture0_0.json"
    else:
        json_file =  f"{COMPARE_DATA_SET_PATH}/Capture1_0.json"
        json_file_write = f"{DATA_SET_PATH}/dataset/Capture1_0.json"


    json_data = rw_json_data(READ, json_file, None)

    read_floder = f"{DATA_SET_PATH}/dataset/"
    CAPTURE_DATA = []
    if LR == 0:
        cam_image_files = sorted(glob.glob(read_floder + 'CAM0_*.bmp'))
    else:
        cam_image_files = sorted(glob.glob(read_floder + 'CAM1_*.bmp'))

    # for i in range(len(cam_image_files)):
    #     file_name = f"/data/misc/wmtrace/frame_{i:04}.bmp" 
    #     stream_ts = i
    #     capture_ts = i
    #     CAPTURE_DATA.append({"stream_ts": stream_ts, "capture_ts": capture_ts, "file_name": file_name, "hmd_pos": [0,0,0], "hmd_orient": [0,0,0,1]})

    for i in range(len(cam_image_files)):
        data = json_data['Capture_DATA'][i]
        CAPTURE_DATA.append(data)

    json_data = OrderedDict()
    json_data['Capture_DATA'] = CAPTURE_DATA
    # Write json data
    rw_json_data(WRITE, json_file_write, json_data)

if __name__ == "__main__":
    LR_CNT = 2

    for i in range(LR_CNT):
        mov_to_bmp(i)
        change_file_name(i)
        change_json_data(i)