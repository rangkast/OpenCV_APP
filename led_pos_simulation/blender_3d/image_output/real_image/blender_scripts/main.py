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

# LEGACY
LRVEC = np.array([-1.25370798,  1.12521308,  1.9190501 ] )
LTVEC = np.array([0.02065667, 0.00127346, 0.33140927])

#LRVEC = np.array([-1.30386976,  1.12048586,  1.869319  ]  )
#LTVEC = np.array([ 0.01769808, -0.04910544,  0.33371041])


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
make_camera_follow_path(bpy.data.objects['CAMERA_0_DEFAULT'], bpy.data.objects.get(f"CAMERA_0_FIT_CIRCLE_LOOP"), start_frame=0, end_frame=120)
make_camera_look_at(bpy.data.objects['CAMERA_0_DEFAULT'], bpy.data.objects.get(f"CAMERA_0_FIT_CIRCLE_LOOP_CENTER_TARGET"))
set_camera_roll(bpy.data.objects['CAMERA_0_DEFAULT'], math.radians(90), math.radians(180), math.radians(90)) # adjust roll 90 degrees to the left



'''
    Rendering Images
'''
#save_png_files()
#render_camera_pos_and_png('CAMERA_0_DEFAULT', start_frame=0, end_frame=120, save_pose = 1, do_render=1)
#render_image_inverse('CAMERA_0')

'''
    Show Results
'''



#draw_camera_pos_and_dir_final()

print('################    DONE   ##################')
