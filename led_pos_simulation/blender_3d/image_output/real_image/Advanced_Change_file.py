from Advanced_Function import *    

# files = os.listdir('.')

# for filename in files:
#     if filename.endswith('.png'):
#         new_filename = filename[:-4] + '.bmp'

#         shutil.move(filename, new_filename)








script_dir = os.path.dirname(os.path.realpath(__file__))
# Add the directory containing poselib to the module search path
print(script_dir)

read_floder = f"{script_dir}/../../../../../simulator/openhmd/dataset_20cm_400us/"
write_folder = f"{script_dir}/../../../../../simulator/openhmd/dataset/"
cam_0_image_files = sorted(glob.glob(read_floder + 'CAM0_*.bmp'))

file_name = []
for bmps in cam_0_image_files:
    split_file = bmps.split('/')
    file_name.append(split_file[len(split_file) - 1])



image_files = sorted(glob.glob(os.path.join(script_dir, './render_img/arcturas/test_1/' + '*.png')))
print(len(image_files), ' ', len(cam_0_image_files))

for i, images in enumerate(image_files):
    print(f"{i}, {images}")
    frame = cv2.imread(images)
    cv2.imwrite(f"{write_folder}{file_name[i]}", frame)



# json_file = os.path.join(script_dir, '../../../../../simulator/openhmd/dataset/Capture1_0.json')
# json_data = rw_json_data(READ, json_file, None)

# print(json_data)
# CAPTURE_DATA = []

# for i in range(120):
#     file_name = f"/data/misc/wmtrace/frame_{i:04}.bmp" 
#     stream_ts = i
#     capture_ts = i
#     CAPTURE_DATA.append({"stream_ts": stream_ts, "capture_ts": capture_ts, "file_name": file_name, "hmd_pos": [0,0,0], "hmd_orient": [0,0,0,1]})




# json_data = OrderedDict()
# json_data['Capture_DATA'] = CAPTURE_DATA
# # Write json data
# rw_json_data(WRITE, json_file, json_data)
