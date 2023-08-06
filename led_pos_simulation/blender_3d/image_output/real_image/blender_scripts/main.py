import bpy

# "functions.py" 텍스트를 모듈로 가져옵니다.
functions = bpy.data.texts["functions.py"].as_module()
# definitions 모듈 내의 모든 이름을 가져옵니다.
names = dir(functions)
# 모든 이름에 대해 전역 변수를 설정합니다.
globals().update({name: getattr(functions, name) for name in names})

'''
Rendering Configuration
'''
# 카메라 해상도 설정 (예: 1920x1080)
bpy.context.scene.render.resolution_x = 1280
bpy.context.scene.render.resolution_y = 960

# 렌더링 결과의 픽셀 밀도를 100%로 설정 (기본값은 50%)
bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.render.film_transparent = False  # 렌더링 배경을 불투명하게 설정
bpy.context.scene.unit_settings.system = 'METRIC'
bpy.context.scene.unit_settings.scale_length = 1.0
bpy.context.scene.unit_settings.length_unit = 'METERS'
#bpy.context.scene.render.engine = 'CYCLES'
# 렌더링 엔진을 Eevee로 설정합니다.
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.eevee.use_bloom = True
# 월드 배경을 어둡게 설정합니다.
set_up_dark_world_background()

'''
TEST START
'''
print('\n\n\n')
print('TEST START')

delete_all_objects_except(exclude_object_names)


make_models('real')


real_camera_data = pickle_data(READ, camera_pickle_file, None)            

#0 frame_cnt 0
#[ 0.40048963 -2.04520374  1.96396611]
#[0.02310081 0.00901199 0.31605864]
#-15 frame_cnt 1
#[ 0.40108061 -1.77605136  2.22209311]
#[0.03932176 0.01539056 0.39029672]
#-30 frame_cnt 2
#[ 0.3958172  -1.47038563  2.44614851]
#[0.05521316 0.04153615 0.46140459]
#15 frame_cnt 3
#[ 0.40819668 -2.27992681  1.66934523]
#[0.00883877 0.02030518 0.24833141]
#data saved

# 15
#LRVEC = np.array([ 0.40819668, -2.27992681,  1.66934523])
#LTVEC = np.array([0.00883877, 0.02030518, 0.24833141])
#start_frame=97, end_frame=217
#math.radians(75), math.radians(0), math.radians(90)
#set reverse

# -15 frame_cnt 1
#LRVEC = np.array([ 0.40108061, -1.77605136,  2.22209311])
#LTVEC = np.array([0.03932176, 0.01539056, 0.39029672])
#start_frame=35, end_frame=155
#math.radians(105), math.radians(180), math.radians(90)

# -30 frame_cnt 2
#LRVEC = np.array([ 0.3958172,  -1.47038563,  2.44614851])
#LTVEC = np.array([0.05521316, 0.04153615, 0.46140459])
#start_frame=35, end_frame=155
#math.radians(120), math.radians(180), math.radians(90)

#0 frame_cnt 0
LRVEC = np.array([ 0.40048963, -2.04520374,  1.96396611])
LTVEC = np.array([0.02310081,0.00901199,0.31605864])
#start_frame=35, end_frame=155
#math.radians(90), math.radians(180), math.radians(90)



make_cameras("CAMERA_0", LRVEC, LTVEC, cam_0_matrix)
# Make Default Camera
default_rvec_left = np.array([0.0, 0.0, 0.0])
default_tvec_left = np.array([0.0, 0.0, 0.0])
make_cameras("CAMERA_0_DEFAULT", default_rvec_left, default_tvec_left, cam_0_matrix)


'''
    Make Camera Path
'''
#draw_camera_recording('CAMERA_0_DEFAULT')


#custom_camera_tracker('CAMERA_0', 'CAMERA_0_DEFAULT')
#make_camera_follow_path(bpy.data.objects['CAMERA_0_DEFAULT'], bpy.data.objects.get('CAMERA_0_CIRCLE_LOOP'), start_frame=90, end_frame=450)
#make_camera_look_at(bpy.data.objects['CAMERA_0_DEFAULT'], bpy.data.objects.get('CAMERA_0_CIRCLE_LOOP_CENTER_TARGET'))
#set_camera_roll(bpy.data.objects['CAMERA_0_DEFAULT'], math.radians(90), math.radians(0), math.radians(-90)) # adjust roll 90 degrees to the left

fit_circle_tracker('CAMERA_0', 'CAMERA_0_DEFAULT')
make_camera_follow_path(bpy.data.objects['CAMERA_0_DEFAULT'], bpy.data.objects.get(f"CAMERA_0_FIT_CIRCLE_LOOP"), start_frame=35, end_frame=155)
make_camera_look_at(bpy.data.objects['CAMERA_0_DEFAULT'], bpy.data.objects.get(f"CAMERA_0_FIT_CIRCLE_LOOP_CENTER_TARGET"))
# default 90 180 90
set_camera_roll(bpy.data.objects['CAMERA_0_DEFAULT'], math.radians(90), math.radians(180), math.radians(90))


'''
    Rendering Images
'''
#save_png_files()
#render_camera_pos_and_png('CAMERA_0_DEFAULT', start_frame=35, end_frame=155, save_pose = 1, do_render=0, do_reverse=0)
#render_image_inverse('CAMERA_0')

'''
    Show Results
'''



#draw_camera_pos_and_dir_final()

print('################    DONE   ##################')
