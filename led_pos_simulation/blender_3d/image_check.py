from PIL import Image

# 이미지 로딩
img = Image.open("cam_0.png")

# 이미지 정보 보기
print("이미지 크기: ", img.size)
print("이미지 포맷: ", img.format)
print("이미지 모드: ", img.mode)

# 이미지 보기
img.show()
