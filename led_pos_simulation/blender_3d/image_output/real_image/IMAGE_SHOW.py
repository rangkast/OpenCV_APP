from Advanced_Function import *

image_files = sorted(glob.glob(f"{script_dir}/../../../../../dataset/dataset_right_40cm/CAM*.bmp"))
print(f"image_files length {len(image_files)}")
curr_frame_cnt = 0

while True:
     if curr_frame_cnt >= len(image_files):
          break
     frame_0 = cv2.imread(image_files[curr_frame_cnt])
     filename = f"IMAGE Mode {os.path.basename(image_files[curr_frame_cnt])}"
     if frame_0 is None or frame_0.size == 0:
          print(f"Failed to load {image_files[curr_frame_cnt]}, frame_cnt:{curr_frame_cnt}")
          continue

     key = cv2.waitKey(1)
     if key & 0xFF == ord('q'):
          break
     elif key & 0xFF == ord('n'):
          curr_frame_cnt += 1     
     elif key & 0xFF == ord('b'):

          curr_frame_cnt -= 1

     cv2.imshow('IMAGE', frame_0)

cv2.destroyAllWindows()