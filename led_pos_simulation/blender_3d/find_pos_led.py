import bpy
import pickle
import gzip
import numpy as np
import bmesh
import mathutils
import math
import bpy_extras
from mathutils import Vector, geometry
import cv2
import os
from mathutils import Matrix
from bpy_extras import mesh_utils

origin_led_data = np.array([
    [-0.02146761, -0.00343424, -0.01381839],
    [-0.0318701, 0.00568587, -0.01206734],
    [-0.03692925, 0.00930785, 0.00321071],
    [-0.04287211, 0.02691347, -0.00194137],
    [-0.04170018, 0.03609551, 0.01989264],
    [-0.02923584, 0.06186962, 0.0161972],
    [-0.01456789, 0.06295633, 0.03659283],
    [0.00766914, 0.07115411, 0.0206431],
    [0.02992447, 0.05507271, 0.03108736],
    [0.03724313, 0.05268665, 0.01100446],
    [0.04265723, 0.03016438, 0.01624689],
    [0.04222733, 0.0228845, -0.00394005],
    [0.03300807, 0.00371497, 0.00026865],
    [0.03006234, 0.00378822, -0.01297127],
    [0.02000199, -0.00388647, -0.014973]
])


pickle_file = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/find_pos_legacy/result.pickle'
# pickle_file = 'D:/OpenCV_APP/led_pos_simulation/find_pos_legacy/result.pickle'
camera_calibration = {"serial": "WMTD303A5006BW", "camera_f": [714.938, 714.938], "camera_c": [676.234, 495.192], "camera_k": [0.074468, -0.024896, 0.005643, -0.000568]}

with gzip.open(pickle_file, 'rb') as f:
    data = pickle.load(f)

led_data = data['LED_INFO']
# print(data['LED_INFO'])
model_data = data['MODEL_INFO']

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.transparent_max_bounces = 12  # 반사와 굴절 최대 반투명 경계 설정
bpy.context.scene.cycles.preview_samples = 100  # 뷰포트 렌더링 품질 설정

bpy.context.scene.unit_settings.system = 'METRIC'
bpy.context.scene.unit_settings.scale_length = 1.0
bpy.context.scene.unit_settings.length_unit = 'METERS'
 
padding = 0.0  # 원하는 패딩 값을 입력하세요.
# LED 원의 반지름을 설정합니다. 원하는 크기를 입력으로 제공할 수 있습니다.
led_size = 0.002
led_thickness = 0.001

def delete_all_objects_except(exclude_object_names):
    for obj in bpy.data.objects:
        if obj.name not in exclude_object_names:
            bpy.data.objects.remove(obj, do_unlink=True)

def create_mesh_object(coords, name="MeshObject", padding=0.0):
    # 새로운 메시를 생성하고 이름을 설정합니다.
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)

    # 씬에 객체를 추가합니다.
    scene = bpy.context.scene
    scene.collection.objects.link(obj)

    # bmesh를 사용하여 메시를 생성합니다.
    bm = bmesh.new()

    # 좌표에 꼭짓점을 추가합니다.
    verts = [bm.verts.new(coord) for coord in coords]

    # Convex Hull 알고리즘을 사용하여 표면을 생성합니다.
    bmesh.ops.convex_hull(bm, input=verts)

    # 변경 사항을 메시에 적용합니다.
    bm.to_mesh(mesh)
    bm.free()

    # 위아래 패딩을 적용합니다.
    if padding != 0.0:
        bounding_box = obj.bound_box
        z_min = min([point[2] for point in bounding_box])
        z_max = max([point[2] for point in bounding_box])

        # Z 축에만 패딩을 적용하기 위해 Z 좌표를 수정합니다.
        for v in mesh.vertices:
            v.co.z += padding if v.co.z > 0 else -padding

    # 짙은 회색 반투명 재질 생성
    mesh_material = bpy.data.materials.new(name="Mesh_Material")
    mesh_material.use_nodes = True

    # 노드를 수정하여 블렌드 모드 설정
    nodes = mesh_material.node_tree.nodes
    links = mesh_material.node_tree.links

    # 기존 Principled BSDF 노드와 Material Output 노드를 가져옵니다.
    principled_node = nodes.get("Principled BSDF")
    output_node = nodes.get("Material Output")

  # 기존 Principled BSDF 노드의 설정을 변경합니다.
    principled_node.inputs["Base Color"].default_value = (0.1, 0.1, 0.1, 1)  # 짙은 회색
    principled_node.inputs["Alpha"].default_value = 0.5  # 알파 값 변경 (1.0로 설정)
    principled_node.inputs["Transmission"].default_value = 0.5  # 빛 투과값 설정 (1.0로 설정)
    principled_node.inputs["IOR"].default_value = 1.0  # 굴절률 설정 (1.0로 설정)

    # 블렌드 모드 설정
#    mesh_material.blend_method = 'BLEND'  # 블렌드 모드 설정
#    mesh_material.shadow_method = 'NONE'  # 그림자 방법 설정

    # 이제 수정한 Principled BSDF 노드와 Material Output 노드를 연결합니다.
    links.new(principled_node.outputs["BSDF"], output_node.inputs["Surface"])

    # 메시 오브젝트에 반투명 재질 적용
    obj.data.materials.append(mesh_material)

    # 메시 스무딩 설정
    for p in mesh.polygons:
        p.use_smooth = True

    return obj


def create_circle_leds_on_surface(led_coords, led_size, name_prefix="LED"):
    led_objects = []
    distance_to_o = led_size * 0
    for i, coord in enumerate(led_coords):
        # LED 오브젝트의 위치를 조정합니다.
        normalized_direction = Vector(coord).normalized()
        location = [coord[0] - distance_to_o * normalized_direction.x,
                    coord[1] - distance_to_o * normalized_direction.y,
                    coord[2] - distance_to_o * normalized_direction.z]
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=led_size, location=location)
        led_obj = bpy.context.active_object
        led_obj.name = f"{name_prefix}_{i}"

        # 노드 기반의 LED 재질을 설정합니다.
        led_material = bpy.data.materials.new(name=f"LED_Material_{i}")
        led_material.use_nodes = True
        led_material.node_tree.nodes.clear()

        # Emission 쉐이더를 추가합니다.
        nodes = led_material.node_tree.nodes
        links = led_material.node_tree.links
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        emission_node = nodes.new(type='ShaderNodeEmission')

        # Emission 쉐이더의 강도와 색상을 설정합니다.
        emission_node.inputs['Strength'].default_value = 0.03  # 강도 조절
        emission_node.inputs['Color'].default_value = (255, 255, 255, 1)  # 색상 조절

        # Emission 쉐이더를 출력 노드에 연결합니다.
        links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

        led_obj.data.materials.append(led_material)
        
        led_objects.append(led_obj)

    return led_objects



def create_circle_leds_with_light_on_surface(led_coords, led_size, mesh_obj, led_power=100, name_prefix="LED"):
    led_objects = create_circle_leds_on_surface(led_coords, led_size, name_prefix)
    
    for i, led_obj in enumerate(led_objects):
        # 빛을 생성합니다.
        light_data = bpy.data.lights.new(name=f"Light_{i}", type='POINT')
        light_data.energy = led_power

        # 빛을 LED 오브젝트에 넣습니다.
        light_obj = bpy.data.objects.new(name=f"Light_{i}", object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        
        # 빛을 LED 위치로 이동합니다.
        light_obj.location = led_obj.location

        # 빛을 메시 오브젝트 표면의 조금 안쪽으로 옮깁니다.
        light_obj.location -= (mesh_obj.location - led_obj.location).normalized() * 0.01

    return led_objects



def set_up_eevee_render_engine():
    # 렌더링 엔진을 Eevee로 설정합니다.
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    # Eevee 렌더링 설정을 조절합니다.
    bpy.context.scene.eevee.use_bloom = True
    bpy.context.scene.eevee.bloom_threshold = 0.8
    bpy.context.scene.eevee.bloom_radius = 6.5
    bpy.context.scene.eevee.bloom_intensity = 0.1

def set_up_dark_world_background():
    # 월드 배경을 어둡게 설정합니다.
    world = bpy.context.scene.world
    world.use_nodes = True
    bg_node = world.node_tree.nodes["Background"]
    bg_node.inputs["Color"].default_value = (0, 0, 0, 1)  # 검은색 배경
    bg_node.inputs["Strength"].default_value = 0.5


# Rotation Matrix to Euler Angles (XYZ)
def rotation_matrix_to_euler_angles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def rotation_matrix_to_quaternion(R):
    qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)

    return np.array([qw, qx, qy, qz])


def calculate_camera_position_direction(rvec, tvec):
    # Extract rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    R_inv = np.linalg.inv(R)
    # Camera position (X, Y, Z)
    Cam_pos = -R_inv @ tvec
    X, Y, Z = Cam_pos.ravel()
    unit_z = np.array([0, 0, 1])

    # euler_angles = rotation_matrix_to_euler_angles(R)
    # print("Euler angles (in degrees):", euler_angles)

    # idea 1
    # cosine_for_pitch = math.sqrt(R[0][0] ** 2 + R[1][0] ** 2)
    # is_singular = cosine_for_pitch < 10**-6
    # if not is_singular:
    #     print('not singular')
    #     roll = math.atan2(R[2][1], R[2][2])
    #     pitch = math.atan2(-R[2][0], math.sqrt(R[2][1]**2 + R[2][2]**2))
    #     yaw = math.atan2(R[1][0], R[0][0])
    # else:
    #     print('singular')
    #     yaw = math.atan2(-R[1][2], R[1][1])
    #     pitch = math.atan2(-R[2][0], cosine_for_pitch)
    #     roll = 0

    # idea 2
    Zc = np.reshape(unit_z, (3, 1))
    Zw = np.dot(R_inv, Zc)  # world coordinate of optical axis
    zw = Zw.ravel()

    pan = np.arctan2(zw[1], zw[0]) - np.pi / 2
    tilt = np.arctan2(zw[2], np.sqrt(zw[0] * zw[0] + zw[1] * zw[1]))

    # roll
    unit_x = np.array([1, 0, 0])
    Xc = np.reshape(unit_x, (3, 1))
    Xw = np.dot(R_inv, Xc)  # world coordinate of camera X axis
    xw = Xw.ravel()
    xpan = np.array([np.cos(pan), np.sin(pan), 0])

    roll = np.arccos(np.dot(xw, xpan))  # inner product
    if xw[2] < 0:
        roll = -roll

    roll = math.degrees(roll)
    pan = math.degrees(pan)
    tilt = math.degrees(tilt)

    print('degrees', 'roll', roll, 'pan', pan, 'tilt', tilt)

    optical_axis = R.T @ unit_z.T
    # 카메라 위치에서 optical axis까지의 방향 벡터 계산
    optical_axis_x, optical_axis_y, optical_axis_z = optical_axis

    return (X, Y, Z), (optical_axis_x, optical_axis_y, optical_axis_z), (roll, pan, tilt)

def create_camera(camera_f, camera_c, cam_location, direction, name, rot):
    roll = math.radians(-rot[0])
    pitch = math.radians(rot[1])
    yaw = math.radians(90 + rot[2])


    X, Y, Z = cam_location
    print('pos', X, Y, Z)
    # Remove existing camera
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
    
    bpy.ops.object.camera_add(location=(X, Y, Z))
    cam = bpy.context.active_object
    cam.name = name
    
    # idea 1    
    # Create a new camera object
    '''
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)  # Normalize direction vector
    rot_quat = mathutils.Vector(direction).to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    '''
    # idea 2    
    cam.rotation_euler = (yaw, roll, pitch)
        
    # idea 3
    '''
    X, Y, Z = cam_location
    # 월드 좌표계에서의 Roll, Pitch, Yaw (예시)
    world_roll, world_pitch, world_yaw = roll, pitch, yaw
    # 축 변환: 월드 좌표계 -> 블렌더 좌표계
    blender_roll = -world_pitch
    blender_pitch = world_roll
    blender_yaw = world_yaw

    # 카메라 방향 설정
    cam.rotation_euler = (blender_roll, blender_pitch, blender_yaw)
    '''
    # idea 4
    '''
    # 월드 좌표계에서의 카메라 위치 (예시)

    # 월드 좌표계에서의 Roll, Pitch, Yaw (예시)
    world_roll, world_pitch, world_yaw = roll, pitch, yaw
    # 월드 좌표계 -> 블렌더 좌표계 변환 행렬
    conversion_matrix = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    # 월드 좌표계에서의 회전 행렬 생성
    R_world = np.array([
        [np.cos(world_yaw) * np.cos(world_pitch), np.cos(world_yaw) * np.sin(world_pitch) * np.sin(world_roll) - np.sin(world_yaw) * np.cos(world_roll), np.cos(world_yaw) * np.sin(world_pitch) * np.cos(world_roll) + np.sin(world_yaw) * np.sin(world_roll)],
        [np.sin(world_yaw) * np.cos(world_pitch), np.sin(world_yaw) * np.sin(world_pitch) * np.sin(world_roll) + np.cos(world_yaw) * np.cos(world_roll), np.sin(world_yaw) * np.sin(world_pitch) * np.cos(world_roll) - np.cos(world_yaw) * np.sin(world_roll)],
        [-np.sin(world_pitch), np.cos(world_pitch) * np.sin(world_roll), np.cos(world_pitch) * np.cos(world_roll)]
    ])
    # 회전 행렬을 블렌더 좌표계로 변환
    R_blender = conversion_matrix @ R_world @ np.linalg.inv(conversion_matrix)
    # 변환된 회전 행렬로부터 오일러 각 추출
    blender_euler = np.array(np.linalg.inv(conversion_matrix) @ R_blender @ conversion_matrix).ravel()
    # 카메라 방향 설정
    cam.rotation_mode = 'XYZ'
    cam.rotation_euler = blender_euler
    '''  
    
    fx_px, fy_px = camera_f
    cx_px, cy_px = camera_c

    # Set the camera sensor size in pixels
    sensor_width_px = 1280.0
    sensor_height_px = 960.0

    # Calculate the sensor size in millimeters
    sensor_width_mm = 36.0 # Assuming 35mm camera sensor size
    sensor_height_mm = 27.0 # Assuming 35mm camera sensor size
    scale = sensor_width_px / sensor_width_mm # Pixel per mm scale factor
    sensor_width = sensor_width_px / scale
    sensor_height = sensor_height_px / scale

    # Calculate the focal length in millimeters
    fx_mm = fx_px / scale
    fy_mm = fy_px / scale
    focal_length = (fx_mm + fy_mm) / 2.0

    # Calculate the image center in pixels
    cx = sensor_width_px / 2.0 + cx_px / scale
    cy = sensor_height_px / 2.0 - cy_px / scale


    # Set the camera parameters
    cam.data.type = 'PERSP'
    cam.data.lens_unit = 'FOV'
    cam.data.angle = 2 * math.atan(sensor_width / (2 * focal_length)) # Field of view in radians
    cam.data.sensor_width = sensor_width
    cam.data.sensor_height = sensor_height
    cam.data.shift_x = (cx - sensor_width_px / 2.0) / fx_px # Shift in X direction
    cam.data.shift_y = (cy - sensor_height_px / 2.0) / fy_px # Shift in Y direction

    return cam


def plot_camera(position, direction, axis_length=1.0, save_path='./camera_plot.png'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    origin = np.array(position)
    end_point = origin + axis_length * np.array(direction)

    ax.quiver(origin[0], origin[1], origin[2],
              direction[0], direction[1], direction[2],
              length=axis_length, color='r')

    ax.set_xlim(origin[0] - 5, origin[0] + 5)
    ax.set_ylim(origin[1] - 5, origin[1] + 5)
    ax.set_zlim(origin[2] - 5, origin[2] + 5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Save the plot to a file
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {os.path.abspath(save_path)}")

def create_point(location, point_name="Point"):
    mesh = bpy.data.meshes.new(point_name)
    obj = bpy.data.objects.new(point_name, mesh)

    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bm = bmesh.new()
    bmesh.ops.create_vert(bm, co=location)
    bm.to_mesh(mesh)
    bm.free()

    return obj

def create_direction_point(start_location, direction, point_name="DirectionPoint", scale=1.0, color=(1, 0, 0, 1)):
    mesh = bpy.data.meshes.new(point_name)
    obj = bpy.data.objects.new(point_name, mesh)

    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bm = bmesh.new()
    end_vert = bm.verts.new(mathutils.Vector(start_location) + mathutils.Vector(direction) * scale)

    bm.to_mesh(mesh)
    bm.free()

    # Set material and color
    mat = bpy.data.materials.new(name="DirectionPointMat")
    mat.diffuse_color = color
    obj.data.materials.append(mat)

    return obj



def make_cameras(camera_name, rvec, tvec, camera_matrix):
    position, direction, rot = calculate_camera_position_direction(rvec, tvec)

    # print(position, direction)

    point_obj = create_point(position)
    direction_point_obj = create_direction_point(position, direction, scale=2.0)

    camera = create_camera(camera_matrix['camera_f'], camera_matrix['camera_c'], position, direction, camera_name, rot)    
#    plot_camera(position, direction, save_path='./camera_plot.png')


def draw_line(p1, p2, name):
    p1 = Vector(p1)
    p2 = Vector(p2)
    direction = p2 - p1
    scaled_direction = direction.normalized() * direction.length * 10
    p2 = p1 + scaled_direction

    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.0001, depth=(p1 - p2).length, end_fill_type='NGON', location=(0,0,0), scale=(1, 1, 1))
    line = bpy.context.object
    line.name = name
    line.location = (p1 + p2) / 2
    line.rotation_mode = 'QUATERNION'
    line.rotation_quaternion = (p1 - p2).to_track_quat('Z', 'Y')


def create_sphere(location, radius, name):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius, location=location)
    sphere = bpy.context.object
    sphere.name = name

def find_intersection(p1, p2, obj):
    intersections = []
    for face in obj.data.polygons:
        vertices = [obj.matrix_world @ obj.data.vertices[v].co for v in face.vertices]
        intersection = geometry.intersect_ray_tri(vertices[0], vertices[1], vertices[2], (p2 - p1), p1, True)
        if intersection:
            intersections.append(intersection)

    return intersections


# Eevee 렌더링 엔진 설정을 적용합니다.
set_up_eevee_render_engine()

# 월드 배경을 어둡게 설정합니다.
set_up_dark_world_background()

# delte objects
exclude_object_names = ["Oculus_L_05.002"]
delete_all_objects_except(exclude_object_names)

# Simulator
'''
# numpy 배열로부터 좌표 데이터를 가져옵니다.
model_data = np.array(model_data)
# led_data를 numpy 배열로 변환합니다.
led_data = np.array(led_data)

# 이 함수 호출로 메시 객체를 생성하고 표면을 그립니다.
create_mesh_object(model_data, padding=padding)
# 모델 오브젝트를 찾습니다.
model_obj = bpy.data.objects['MeshObject']
create_circle_leds_on_surface(led_data, led_size)
'''
create_circle_leds_on_surface(origin_led_data, led_size)


origin = np.array([0, 0, 0])

# 이미 생성된 오브젝트를 이름으로 찾습니다.
torus = bpy.data.objects.get('Oculus_L_05.002')


for idx, led in enumerate(origin_led_data):
    p1 = np.array(origin)
    p2 = np.array(led)
    draw_line(p1, p2, f"LED_Line_{idx + 1}")

    # intersections = find_intersection(Vector(p1), Vector(p2), torus)
    # if intersections:
    #     create_sphere(intersections[0], 0.002, f"Intersection_Sphere_{idx + 1}")
        
default_cameraK = {'serial': "default", 'camera_f': [1, 1], 'camera_c': [0, 0]}
cam_0_matrix = {'serial': "WMTD306N100AXM", 'camera_f': [712.623, 712.623], 'camera_c': [653.448, 475.572]}
cam_1_matrix = {'serial': "WMTD305L6003D6", 'camera_f': [716.896, 716.896], 'camera_c': [668.902, 460.618]}

#camera_matrix = np.array([[camera_calibration['camera_f'][0], 0, camera_calibration['camera_c'][0]], [0, camera_calibration['camera_f'][1], camera_calibration['camera_c'][1]], [0, 0, 1]])

# left
rvec_left = np.array([ 0.92258546, -0.31966116,  1.53042547])
tvec_left = np.array([-0.10226342,  0.21217596,  0.45900419])

# right
rvec_right = np.array([ 0.82721212, -1.71101202,  0.74346322])
tvec_right = np.array([-0.08838871,  0.08055683, 0.446927  ])

# 카메라 해상도 설정 (예: 1920x1080)
bpy.context.scene.render.resolution_x = 1280
bpy.context.scene.render.resolution_y = 960

# 렌더링 결과의 픽셀 밀도를 100%로 설정 (기본값은 50%)
bpy.context.scene.render.resolution_percentage = 100

make_cameras("CAMERA_0", rvec_left, tvec_left, cam_0_matrix)
make_cameras("CAMERA_1", rvec_right, tvec_right, cam_1_matrix)