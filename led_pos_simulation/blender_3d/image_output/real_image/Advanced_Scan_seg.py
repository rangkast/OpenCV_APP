import numpy as np

def find_islands_scan(img, threshold):
    rows, cols = img.shape

    def check_islands(sequence):
        # threshold 처리
        sequence = [x if x > threshold else 0 for x in sequence]
        print(f"sequence { sequence}")
        falling_edge = 0
        rising_edge = 0
        prev_data = 0
        for data in sequence:
            if prev_data == 0 and data != 0:
                rising_edge += 1
            if prev_data !=0 and data == 0:
                falling_edge += 1          
            prev_data = data
            if rising_edge >= 2 or falling_edge >=2:
                return True

        return False

    for i in range(rows):
        if check_islands(img[i, :]):  # 각 행에 대해서 확인합니다.
            return -1

    for j in range(cols):
        if check_islands(img[:, j]):  # 각 열에 대해서 확인합니다.
            return -1

    return 0

# 테스트
img = np.array([
    [10, 30, 160, 114, 114, 170, 0],
    [10, 255, 114, 114, 114, 100, 0],
    [10, 30, 100, 114, 114, 170, 0],
    [10, 30, 100, 114, 114, 170, 0]
], dtype=np.uint8)

result = find_islands_scan(img, 100)
print(f"Result: {result}")  # 출력: Result: -1
