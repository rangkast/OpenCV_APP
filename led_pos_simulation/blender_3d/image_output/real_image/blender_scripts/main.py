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

# delte objects
exclude_object_names = [
#                        "CAMERA_0",
#                        "CAMERA_1",
#                        "CAMERA_0_DEFAULT",
                        "Oculus_L_05.002",
#                        "EMPTY_CAMERA_0",
#                        "EMPTY_CAMERA_1",
                        "sine_wave",
                        "circle_curve",
                        "quad_circle_curve_2_45",
                        "quad_circle_curve_3_45",
                        "axis_circle",
                        "robot_circle",
                        "custom_circle",
                        "RELATIVE_PATH"
                        ]
delete_all_objects_except(exclude_object_names)


make_models('real')
make_cameras_model()


'''
    Make Camera Path
'''
#draw_camera_recording('CAMERA_0_DEFAULT')
#custom_camera_tracker('CAMERA_0', 'CAMERA_0_DEFAULT')


'''
    Attach Camera to Path
'''
#make_camera_follow_path(bpy.data.objects['CAMERA_0_DEFAULT'], bpy.data.objects.get('robot_circle'))
#make_camera_look_at(bpy.data.objects['CAMERA_0_DEFAULT'], bpy.data.objects.get('Controller'))
#set_camera_roll(bpy.data.objects['CAMERA_0_DEFAULT'], math.radians(90), math.radians(0), math.radians(-90))
make_camera_follow_path(bpy.data.objects['CAMERA_0_DEFAULT'], bpy.data.objects.get('CAMERA_0_CIRCLE_LOOP'), start_frame=90, end_frame=450)
make_camera_look_at(bpy.data.objects['CAMERA_0_DEFAULT'], bpy.data.objects.get('CAMERA_0_CIRCLE_LOOP_CENTER_TARGET'))
set_camera_roll(bpy.data.objects['CAMERA_0_DEFAULT'], math.radians(90), math.radians(0), math.radians(-90)) # adjust roll 90 degrees to the left


'''
    Rendering Images
'''
#save_png_files()
#render_camera_pos_and_png('CAMERA_0_DEFAULT', start_frame=90, end_frame=450)
#render_image_inverse('CAMERA_0')


'''
    Show Results
'''
print('Show Test Results')

print('################    DONE   ##################')
