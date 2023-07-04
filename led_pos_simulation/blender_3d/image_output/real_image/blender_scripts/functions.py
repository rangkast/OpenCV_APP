import bpy
import bgl
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
from math import degrees
from datetime import datetime
import platform
import json
from collections import OrderedDict
from scipy.spatial.transform import Rotation as Rot
import random
from mathutils import Quaternion

READ = 0
WRITE = 1

ERROR = 'ERROR'
SUCCESS = 'SUCCESS'

DONE = 'DONE'
NOT_SET = 'NOT_SET'


'''
Definitions
'''
camera_names = ["CAMERA_0", "CAMERA_1"]

padding = 0.0  # 원하는 패딩 값을 입력하세요.
# LED 원의 반지름을 설정합니다. 원하는 크기를 입력으로 제공할 수 있습니다.
#led_size = 0.0025
led_size = 0.0025
led_thickness = 0.0022
emission_strength = 0.01


default_cameraK = {'serial': "default", 'camera_f': [1, 1], 'camera_c': [0, 0]}
default_dist_coeffs = np.zeros((4, 1))
cam_0_matrix = {'serial': "WMTD306N100AXM", 'camera_f': [712.623, 712.623], 'camera_c': [653.448, 475.572], 'dist_coeff': np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)}
cam_1_matrix = {'serial': "WMTD305L6003D6", 'camera_f': [716.896, 716.896], 'camera_c': [668.902, 460.618], 'dist_coeff': np.array([[0.07542], [-0.026874], [0.006662], [-0.000775]], dtype=np.float64)}


'''
Simulation shampe
'''
# shape = 'plane'
# shape = 'sphere'
# shape = 'cylinder'
#shape = 'cylinder_base'
#shape = 'basic'

'''
Real UVC Camera
'''
shape = 'real'

MESH_OBJ_NAME = 'MeshObject_' + f'{shape}'


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



model_pickle_file = None
camera_pickle_file = None
os_name = platform.system()
if os_name == 'Windows':
    print("This is Windows")
    if shape == 'sphere':
        model_pickle_file = 'D:/OpenCV_APP/led_pos_simulation/find_pos_legacy/result.pickle'
    elif shape == 'cylinder':
        model_pickle_file = 'D:/OpenCV_APP/led_pos_simulation/find_pos_legacy/result_cylinder.pickle'
    elif shape == 'cylinder_base':
        model_pickle_file = 'D:/OpenCV_APP/led_pos_simulation/find_pos_legacy/result_cylinder_base.pickle'
        base_file_path = 'D:/OpenCV_APP/led_pos_simulation/blender_3d/image_output/cylinder_base/'
    elif shape == 'real':
        base_file_path = 'D:/OpenCV_APP/led_pos_simulation/blender_3d/image_output/real_image/'
    else:
        model_pickle_file = 'D:/OpenCV_APP/led_pos_simulation/find_pos_legacy/basic_test.pickle'
        base_file_path = 'D:/OpenCV_APP/led_pos_simulation/blender_3d/image_output/blender_basic/'
    
    camera_pickle_file = 'D:/OpenCV_APP/led_pos_simulation/blender_3d/real_camera_data.pickle'
    camera_info_path = "D:/OpenCV_APP/led_pos_simulation/blender_3d/image_output/real_image/rt_std.pickle"
    ba_result_path = "D:/OpenCV_APP/led_pos_simulation/blender_3d/image_output/real_image/bundle.pickle"
    render_folder = 'D:/OpenCV_APP/led_pos_simulation/blender_3d/image_output/real_image/tmp/render'
elif os_name == 'Linux':
    print("This is Linux")
    if shape == 'sphere':
        model_pickle_file = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/find_pos_legacy/result.pickle'
    elif shape == 'cylinder':
        model_pickle_file = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/find_pos_legacy/result_cylinder.pickle'
    elif shape == 'cylinder_base':
        model_pickle_file = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/find_pos_legacy/result_cylinder_base.pickle'
        base_file_path = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/blender_3d/image_output/cylinder_base/'
    elif shape == 'real':
        base_file_path = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/blender_3d/image_output/real_image/'
    else:
        model_pickle_file = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/find_pos_legacy/basic_test.pickle'
        base_file_path = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/blender_3d/image_output/blender_basic/'
    
    camera_pickle_file = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/blender_3d/real_camera_data.pickle'
    camera_info_path = "/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/blender_3d/image_output/real_image/rt_std.pickle"
    ba_result_path = "/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/blender_3d/image_output/real_image/bundle.pickle"
    render_folder = "/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/blender_3d/image_output/real_image/tmp/render/"
else:
    print("Unknown OS")

image_file_path = base_file_path
blend_file_path = base_file_path + 'blender_test_image.blend'


## origin
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

origin_led_dir = np.array([
    [-0.52706841, -0.71386452, -0.46108171],
    [-0.71941994, -0.53832866, -0.43890456],
    [-0.75763735, -0.6234486, 0.19312559],
    [-0.95565641, 0.00827838, -0.29436762],
    [-0.89943476, -0.04857372, 0.43434745],
    [-0.57938915, 0.80424722, -0.13226727],
    [-0.32401356, 0.5869508, 0.74195955],
    [0.14082806, 0.97575588, -0.16753482],
    [0.66436362, 0.41503629, 0.62158335],
    [0.77126662, 0.61174447, -0.17583089],
    [0.90904575, -0.17393345, 0.37865945],
    [0.9435189, -0.10477919, -0.31431419],
    [0.7051038, -0.6950803, 0.14032818],
    [0.67315478, -0.5810967, -0.45737213],
    [0.49720891, -0.70839529, -0.5009585]
])

# Set the seed for Python's random module.
random.seed(1)
# Set the seed for NumPy's random module.
np.random.seed(1)
noise_std_dev = 0.0015  # Noise standard deviation. Adjust this value to your needs.
# Generate noise with the same shape as the original data.
noise = np.random.normal(scale=noise_std_dev, size=origin_led_data.shape)
# Add noise to the original data.
target_led_data = origin_led_data + noise
 
 
 # 이동 벡터 정의
translation_vector = np.array([0.01, 0, 0])

# 각 축에 대한 회전 각도 정의
rotation_degrees_x = 5
rotation_degrees_y = 5
rotation_degrees_z = 0

# 각 축에 대한 회전 객체 생성
rotation_x = Rot.from_rotvec(rotation_degrees_x / 180.0 * np.pi * np.array([1, 0, 0]))
rotation_y = Rot.from_rotvec(rotation_degrees_y / 180.0 * np.pi * np.array([0, 1, 0]))
rotation_z = Rot.from_rotvec(rotation_degrees_z / 180.0 * np.pi * np.array([0, 0, 1]))

# LED 좌표 이동 및 회전
new_led_data = np.empty_like(target_led_data)
new_led_dir = np.empty_like(origin_led_dir)
for i in range(len(target_led_data)):
    # 이동 적용
    new_led_data[i] = target_led_data[i] + translation_vector
    # 회전 적용
    new_led_data[i] = rotation_x.apply(new_led_data[i])
    new_led_data[i] = rotation_y.apply(new_led_data[i])
    new_led_data[i] = rotation_z.apply(new_led_data[i])

    new_led_dir[i] = rotation_x.apply(origin_led_dir[i])
    new_led_dir[i] = rotation_y.apply(new_led_dir[i])
    new_led_dir[i] = rotation_z.apply(new_led_dir[i])


offset = 0.009593
new_led_data[:, 2] -= offset


print('#############################')
print("target_led_data:\n", target_led_data)
print("New LED coordinates:\n", new_led_data)
print("New LED directions:\n", new_led_dir)

origin_led_data = new_led_data
origin_led_dir = new_led_dir



## rifts3
#origin_led_data = np.array([
#    [-0.02164717, -0.00339185, -0.01388412],
#    [-0.03201647, 0.00567471, -0.01216926],
#    [-0.0370172, 0.00917934, 0.00318938],
#    [-0.04297077, 0.0270956, -0.00194208],
#    [-0.04176234, 0.03597212, 0.02000342],
#    [-0.02931414, 0.06181315, 0.01623857],
#    [-0.01453551, 0.06311053, 0.03652011],
#    [0.007677, 0.07117793, 0.02074927],
#    [0.03000516, 0.05517989, 0.03113699],
#    [0.03722962, 0.05273691, 0.01088159],
#    [0.04276992, 0.03016744, 0.01631011],
#    [0.04226804, 0.02280077, -0.00386487]
#    [0.033007, 0.00368969, 0.00022582],
#    [0.03026651, 0.00387004, -0.01306987],
#    [0.02019107, -0.0041028, -0.01489263],
#])
#origin_led_data = np.array([
#    [-0.52706841, -0.71386452, -0.46108171],
#    [-0.71941994, -0.53832866, -0.43890456],
#    [-0.75763735, -0.6234486, 0.19312559],
#    [-0.95565641, 0.00827838, -0.29436762],
#    [-0.89943476, -0.04857372, 0.43434745],
#    [-0.57938915, 0.80424722, -0.13226727],
#    [-0.32401356, 0.5869508, 0.74195955],
#    [0.14082806, 0.97575588, -0.16753482],
#    [0.66436362, 0.41503629, 0.62158335],
#    [0.77126662, 0.61174447, -0.17583089],
#    [0.90904575, -0.17393345, 0.37865945],
#    [0.9435189, -0.10477919, -0.31431419],
#    [0.7051038, -0.6950803, 0.14032818],
#    [0.67315478, -0.5810967, -0.45737213],
#    [0.49720891, -0.70839529, -0.5009585],
#])


'''
Functions
'''

def delete_all_objects_except(exclude_object_names):
    excluded_objects = set()

    for name in exclude_object_names:
        # If the name is an object, add it to the excluded_objects set
        if name in bpy.data.objects:
            excluded_objects.add(name)
        # If the name is a collection, add all objects in the collection to the excluded_objects set
        elif name in bpy.data.collections:
            for obj in bpy.data.collections[name].objects:
                excluded_objects.add(obj.name)

    for obj in bpy.data.objects:
        if obj.name not in excluded_objects:
            bpy.data.objects.remove(obj, do_unlink=True)



def create_mesh_object(coords, name, padding=0.0):
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
    principled_node.inputs["Alpha"].default_value = 1.0  # 알파 값 변경 (1.0로 설정)
    principled_node.inputs["Transmission"].default_value = 1.0  # 빛 투과값 설정 (1.0로 설정)
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


def create_circle_leds_on_surface(led_coords, led_size, shape, name_prefix="LED"):
    led_objects = []

    for i, coord in enumerate(led_coords):
        # LED 오브젝트의 위치를 조정합니다.
        normalized_direction = Vector(coord).normalized()
        if shape == 'sphere':
            distance_to_o = led_size * 2 / 3
            location = [coord[0] - distance_to_o * normalized_direction.x,
                        coord[1] - distance_to_o * normalized_direction.y,
                        coord[2] - distance_to_o * normalized_direction.z]
        else:
            location = coord

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
        emission_node.inputs['Strength'].default_value = emission_strength  # 강도 조절
        emission_node.inputs['Color'].default_value = (255, 255, 255, 1)  # 색상 조절

        # Emission 쉐이더를 출력 노드에 연결합니다.
        links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])

        led_obj.data.materials.append(led_material)

        led_objects.append(led_obj)

    return led_objects


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




def DeselectEdgesAndPolygons(obj):
    for p in obj.data.polygons:
        p.select = False
    for e in obj.data.edges:
        e.select = False


def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in 
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0  # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
         (0, s_v, v_0),
         (0, 0, 1)))

    print('sensor_fit', sensor_fit)
    print('f_in_mm', f_in_mm)
    print('sensor_size_in_mm', sensor_size_in_mm)
    print('res_x res_y', resolution_x_in_px, resolution_y_in_px)
    print('pixel_aspect_ratio', pixel_aspect_ratio)
    print('shift_x shift_y', camd.shift_x, camd.shift_y)
    print('K', K)

    return K


# Returns camera rotation and translation matrices from Blender.
# 
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates 
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
    ))
    return RT


def get_3x4_RT_matrix_from_blender_OpenCV(obj):
    isCamera = (obj.type == 'CAMERA')
    R_BlenderView_to_OpenCVView = np.diag([1 if isCamera else -1, -1, -1])
#    print('R_BlenderView_to_OpenCVView', R_BlenderView_to_OpenCVView)
    location, rotation = obj.matrix_world.decompose()[:2]
#    print('location', location, 'rotation', rotation)
    # Convert the rotation to axis-angle representation
    axis, angle = rotation.to_axis_angle()

    # Create a 3x3 rotation matrix from the axis-angle representation
    R_BlenderView = Matrix.Rotation(angle, 3, axis).transposed()

    T_BlenderView = -1.0 * R_BlenderView @ location

    R_OpenCV = R_BlenderView_to_OpenCVView @ R_BlenderView
    T_OpenCV = R_BlenderView_to_OpenCVView @ T_BlenderView

    R, _ = cv2.Rodrigues(R_OpenCV)
#    print('R_OpenCV', R_OpenCV)
#    print('R_OpenCV(Rod)', R.ravel())
#    print('T_OpenCV', T_OpenCV)

    RT_OpenCV = Matrix(np.column_stack((R_OpenCV, T_OpenCV)))
    return RT_OpenCV, R_OpenCV, T_OpenCV


def get_3x4_P_matrix_from_blender_OpenCV(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender_OpenCV(cam)[0]
    return K @ RT


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K @ RT, K, RT



def blender_location_rotation_from_opencv(rvec, tvec, isCamera=True):
    R_BlenderView_to_OpenCVView = Matrix([
        [1 if isCamera else -1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ])

    # Convert rvec to rotation matrix
    R_OpenCV, _ = cv2.Rodrigues(rvec)

    # Convert OpenCV R|T to Blender R|T
    R_BlenderView = R_BlenderView_to_OpenCVView @ Matrix(R_OpenCV.tolist())
    T_BlenderView = R_BlenderView_to_OpenCVView @ Vector(tvec)

    # Invert rotation matrix
    R_BlenderView_inv = R_BlenderView.transposed()

    # Calculate location
    location = -1.0 * R_BlenderView_inv @ T_BlenderView

    # Convert rotation matrix to quaternion
    rotation = R_BlenderView_inv.to_quaternion()

    return location, rotation


# ----------------------------------------------------------
# Alternate 3D coordinates to 2D pixel coordinate projection code
# adapted from https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex?lq=1
# to have the y axes pointing up and origin at the top-left corner
def project_by_object_utils(cam, point):
    scene = bpy.context.scene
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
    print('co_2d', co_2d)
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    return (co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1])


def quaternion_to_euler_degree(quaternion):
    # Convert quaternion to Euler rotation (radians)
    euler_rad = quaternion.to_euler()

    # Convert radians to degrees
    euler_deg = Vector([math.degrees(axis) for axis in euler_rad])

    return euler_deg


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


def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


#def create_camera(camera_f, camera_c, cam_location, name, rot):
#    rotation = quaternion_to_euler_degree(rot)
#    print('euler(degree)', rotation)

#    X, Y, Z = cam_location
#    print('position', X, Y, Z)

#    if name in bpy.data.objects:
#        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

#    bpy.ops.object.camera_add(location=(X, Y, Z))

#    cam = bpy.context.active_object
#    cam.name = name
#    cam.rotation_euler = (math.radians(rotation[0]), math.radians(rotation[1]), math.radians(rotation[2]))
#    
#    fx_px, fy_px = camera_f
#    cx_px, cy_px = camera_c

#    # Set the camera sensor size in pixels
#    sensor_width_px = 1280.0
#    sensor_height_px = 960.0

#    # Calculate the sensor size in millimeters
#    sensor_width_mm = 36  # Assuming 35mm camera sensor size
##    sensor_height_mm = 27.0  # Assuming 35mm camera sensor size
#    scale = sensor_width_px / sensor_width_mm  # Pixel per mm scale factor
#    sensor_width = sensor_width_px / scale
#    sensor_height = sensor_height_px / scale

#    # Calculate the focal length in millimeters
#    fx_mm = fx_px / scale
#    fy_mm = fy_px / scale
#    focal_length = (fx_mm + fy_mm) / 2.0

#    # Calculate the image center in pixels
#    cx = sensor_width_px / 2.0 + cx_px / scale
#    cy = sensor_height_px / 2.0 - cy_px / scale

#    # Set the camera parameters
#    cam.data.type = 'PERSP'
#    cam.data.lens_unit = 'FOV'
#    cam.data.angle = 2 * math.atan(sensor_width / (2 * focal_length))  # Field of view in radians

#    cam.data.sensor_width = sensor_width
#    cam.data.sensor_height = sensor_height

#    scene = bpy.context.scene

#    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x

#    view_fac_in_px = sensor_width_px

#    cam.data.shift_x = (sensor_width_px / 2 - cx_px) / view_fac_in_px
#    cam.data.shift_y = (cy_px - sensor_height_px / 2) * pixel_aspect_ratio / view_fac_in_px

#    print('shift_x, shift_y', cam.data.shift_x, cam.data.shift_y)
#    return cam


def create_camera(camera_f, camera_c, cam_location, name, rot):
    rotation = quaternion_to_euler_degree(rot)
    print('euler(degree)', rotation)

    X, Y, Z = cam_location
    print('position', X, Y, Z)

    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    bpy.ops.object.camera_add(location=(X, Y, Z))

    cam = bpy.context.active_object
    cam.name = name
    cam.rotation_euler = (math.radians(rotation[0]), math.radians(rotation[1]), math.radians(rotation[2]))
    
    fx_px, fy_px = camera_f
    cx_px, cy_px = camera_c

    # Set the camera sensor size in pixels
    sensor_width_px = 1280.0
    sensor_height_px = 960.0

    # Calculate the sensor size in millimeters
    sensor_width_mm = 7.18  # Set camera sensor width to 8mm
    sensor_height_mm = 5.32 # Set camera sensor height to 8mm
    scale = sensor_width_px / sensor_width_mm  # Pixel per mm scale factor
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
    cam.data.lens_unit = 'MILLIMETERS'
    cam.data.lens = focal_length  # Set focal length

    cam.data.sensor_width = sensor_width_mm
    cam.data.sensor_height = sensor_height_mm

    scene = bpy.context.scene
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    view_fac_in_px = sensor_width_px

    cam.data.shift_x = (sensor_width_px / 2 - cx_px) / view_fac_in_px
    cam.data.shift_y = (cy_px - sensor_height_px / 2) * pixel_aspect_ratio / view_fac_in_px

    print('shift_x, shift_y', cam.data.shift_x, cam.data.shift_y)
    return cam



def create_camera_default(cam_location, rot, name):
    # Set the camera sensor size in pixels
    sensor_width_px = 1280.0
    sensor_height_px = 960.0

    rotation = quaternion_to_euler_degree(rot)
    print('euler(degree)', rotation)

    X, Y, Z = cam_location
    print('position', X, Y, Z)
    # Remove existing camera
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    # MAKE DEFAULT CAM
    bpy.ops.object.camera_add(location=(X, Y, Z))
    cam = bpy.context.active_object
    cam.name = name
    cam.rotation_euler = (math.radians(rotation[0]), math.radians(rotation[1]), math.radians(rotation[2]))


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
    # position, direction, rot = calculate_camera_position_direction(rvec, tvec)
    position, rotation = blender_location_rotation_from_opencv(rvec, tvec)
    # print(position, direction)
    # point_obj = create_point(position)
    # direction_point_obj = create_direction_point(position, direction, scale=2.0)
    camera = create_camera(camera_matrix['camera_f'], camera_matrix['camera_c'], position, camera_name, rotation)


def make_cameras_default(camera_name, rvec, tvec):
    position, rotation = blender_location_rotation_from_opencv(rvec, tvec)
    camera = create_camera_default(position, rotation, camera_name)


def export_camera_to_opencv_txt(cam_name, path, file_name):
    cam = bpy.data.objects[cam_name]
    P = get_3x4_P_matrix_from_blender(cam)

    nP = np.matrix(P)
    # path = bpy.path.abspath("//")
    filename = file_name + ".txt"
    file = path + "/" + filename
    np.savetxt(file, nP)
    print(f"Saved to: \"{file}\".")


def export_camera_to_opencv_json(cam_name, path, file_name):
    cam = bpy.data.objects[cam_name]
    _, rvec, tvec = get_3x4_RT_matrix_from_blender(cam)
    R, _ = cv2.Rodrigues(rvec)
    filename = file_name + ".json"
    file = path + "/" + filename
    json_data = OrderedDict()
    json_data['rvec'] = np.array(R.ravel()).tolist()
    json_data['tvec'] = np.array(tvec).tolist()
    rw_json_data(WRITE, file, json_data)
    print(f"Saved to: \"{file}\".")


def rw_json_data(rw_mode, path, data):
    print('json path', path)
    try:
        if rw_mode == 0:
            with open(path, 'r', encoding="utf-8") as rdata:
                json_data = json.load(rdata)
            return json_data
        elif rw_mode == 1:
            with open(path, 'w', encoding="utf-8") as wdata:
                json.dump(data, wdata, ensure_ascii=False, indent="\t")
        else:
            print('not support mode')
    except:
        # print('file r/w error')
        return -1



def draw_line(start, end, color, name):
    m = bpy.data.meshes.new(name)
    o = bpy.data.objects.new(name, m)

    bpy.context.collection.objects.link(o)

    bm = bmesh.new()

    v1 = bm.verts.new(start)
    v2 = bm.verts.new(end)
    e = bm.edges.new([v1, v2])

    bm.to_mesh(m)
    bm.free()

    mat = bpy.data.materials.new(name="line_mat")
    mat.diffuse_color = color
    mat.use_nodes = False
    o.data.materials.append(mat)

    o.hide_render = True  # Hide in rendering
    o.hide_viewport = True
      # Show in viewport



def create_sphere(location, radius, name):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius, location=location)
    sphere = bpy.context.object
    sphere.name = name


def find_intersection(p1, p2, obj, epsilon=1e-8):
    intersections = []
    for face in obj.data.polygons:
        vertices = [obj.matrix_world @ obj.data.vertices[v].co for v in face.vertices]
        intersection = geometry.intersect_ray_tri(vertices[0], vertices[1], vertices[2], (p2 - p1), p1, True)
        if intersection:
            dist_to_intersection = (intersection - p1).length
            dist_p1_to_p2 = (p2 - p1).length
            if abs(dist_to_intersection - dist_p1_to_p2) > epsilon:
                intersections.append(intersection)

    return intersections


def quaternion_to_euler_degree(quaternion):
    # Convert quaternion to Euler rotation (radians)
    euler_rad = quaternion.to_euler()

    # Convert radians to degrees
    euler_deg = Vector([math.degrees(axis) for axis in euler_rad])

    return euler_deg


def apply_boolean_modifier(target_obj, cutter_obj, operation='DIFFERENCE'):
    boolean_mod = target_obj.modifiers.new('Boolean', 'BOOLEAN')
    boolean_mod.operation = operation
    boolean_mod.use_self = True
    boolean_mod.object = cutter_obj

    # 모디파이어를 적용합니다.
    target_obj.select_set(True)
    cutter_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj
    bpy.ops.object.modifier_apply({"object": target_obj}, modifier=boolean_mod.name)

    # 자른 LED 오브젝트를 삭제합니다.
    bpy.ops.object.select_all(action='DESELECT')
    cutter_obj.select_set(True)
    bpy.ops.object.delete()


def create_filled_emission_circle_on_plane(led_coords, origin_led_dir, circle_radius):
    for i, coord in enumerate(led_coords):
        # Calculate normal and rotation quaternion
        normal = origin_led_dir[i]
        rotation_quat = mathutils.Vector((0, 0, 1)).rotation_difference(mathutils.Vector(normal))

        # Adjust the distance to the origin for the inner circle
        distance_to_o = circle_radius * 0.05  
        location_outer = coord
        location_inner = [coord[0] - distance_to_o * normal[0],
                          coord[1] - distance_to_o * normal[1],
                          coord[2] - distance_to_o * normal[2]]

        # Create an outer circle with Emission shader
        bpy.ops.mesh.primitive_circle_add(radius=circle_radius, location=location_outer, fill_type='NGON')
        outer_circle_obj = bpy.context.object

        # Rotate the outer circle to match the plane
        outer_circle_obj.rotation_euler = rotation_quat.to_euler()



#        # Create new material
#        mat_emission = bpy.data.materials.new(name="Emission_Gradient")
#        mat_emission.use_nodes = True

#        nodes = mat_emission.node_tree.nodes
#        links = mat_emission.node_tree.links

#        # Create nodes
#        emission = nodes.new('ShaderNodeEmission')
#        layer_weight = nodes.new('ShaderNodeLayerWeight')
#        color_ramp = nodes.new('ShaderNodeValToRGB')  # Color Ramp node
#        output = nodes['Material Output']

#        # Connect nodes
#        links.new(layer_weight.outputs['Facing'], color_ramp.inputs['Fac'])  # Connect layer weight to color ramp
#        links.new(color_ramp.outputs['Color'], emission.inputs['Color'])  # Connect color ramp to emission
#        links.new(emission.outputs['Emission'], output.inputs['Surface'])
#        # Color Ramp settings
#        color_ramp.color_ramp.elements[0].color = (1, 1, 1, 1)  # white
#        color_ramp.color_ramp.elements[0].position = 0.055  # position of white color
#        color_ramp.color_ramp.elements[1].color = (0.572, 0.572, 0.572, 1)  # black
#        color_ramp.color_ramp.elements[1].position = 0.170  # position of black color
#        color_ramp.color_ramp.elements.new(0.305)  # grey2
#        color_ramp.color_ramp.elements.new(0.517)  # black
#        color_ramp.color_ramp.elements[2].color = (0.438, 0.438, 0.438, 1)  # grey2
#        color_ramp.color_ramp.elements[3].color = (0, 0, 0, 1)  # black
#        color_ramp.color_ramp.interpolation = 'B_SPLINE'  # Set interpolation to B-Spline

#        # Layer Weight settings
#        layer_weight.inputs['Blend'].default_value = 0.5  # adjust this value to control the effect of the layer weight
#        # Emission strength settings
#        emission.inputs['Strength'].default_value = 5  # Set emission strength

        # Add an Emission shader to the outer circle
        mat_emission = bpy.data.materials.new(name="Emission_Material")
        mat_emission.use_nodes = True
        mat_emission.node_tree.nodes.clear()

        nodes = mat_emission.node_tree.nodes
        links = mat_emission.node_tree.links

        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        node_emission = nodes.new(type='ShaderNodeEmission')

        node_emission.inputs['Strength'].default_value = emission_strength
        node_emission.inputs['Color'].default_value = (255, 255, 255, 1)

        links.new(node_emission.outputs['Emission'], node_output.inputs['Surface'])

        outer_circle_obj.data.materials.append(mat_emission)


        # Create an inner circle with Diffuse shader
        bpy.ops.mesh.primitive_circle_add(radius=(circle_radius * 2.0), location=location_inner, fill_type='NGON')
        inner_circle_obj = bpy.context.object

        # Rotate the inner circle to match the plane and point towards origin
        inner_circle_obj.rotation_euler = rotation_quat.to_euler()

        # Add a Diffuse shader to the inner circle
        mat_diffuse = bpy.data.materials.new(name="Diffuse_Material")
        mat_diffuse.use_nodes = True
        mat_diffuse.node_tree.nodes.clear()

        nodes = mat_diffuse.node_tree.nodes
        links = mat_diffuse.node_tree.links

        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        node_diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')

        node_diffuse.inputs['Color'].default_value = (0, 0, 0, 1)  # Black color

        links.new(node_diffuse.outputs['BSDF'], node_output.inputs['Surface'])

        inner_circle_obj.data.materials.append(mat_diffuse)



def create_filled_emission_sphere(led_coords, origin_led_dir, sphere_radius):
    for i, coord in enumerate(led_coords):
        # Calculate normal and rotation quaternion
        normal = origin_led_dir[i]
        rotation_quat = mathutils.Vector((0, 0, 1)).rotation_difference(mathutils.Vector(normal))
        distance_to_o = sphere_radius * 1.0 
        location_inner = [coord[0] - distance_to_o * normal[0],
                          coord[1] - distance_to_o * normal[1],
                          coord[2] - distance_to_o * normal[2]]

#        # Create a sphere with Emission shader
#        bpy.ops.mesh.primitive_uv_sphere_add(radius=sphere_radius, location=coord)
#        sphere_obj = bpy.context.object

        # push
        bpy.ops.mesh.primitive_uv_sphere_add(radius=sphere_radius, location=coord)
        sphere_obj = bpy.context.object

        # Rotate the sphere to match the normal
        sphere_obj.rotation_euler = rotation_quat.to_euler()

        # Squeeze the sphere along the Z-axis
        # You may need to adjust the scale factor to get the desired amount of squeezing
        sphere_obj.scale = (1, 1, 0.2)  # adjust the Z factor to squeeze the sphere

        # Apply scale
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


        # Apply smooth shading
        bpy.ops.object.shade_smooth()
        # Rotate the sphere to match the normal
        sphere_obj.rotation_euler = rotation_quat.to_euler()

        # Create new material
        mat_emission = bpy.data.materials.new(name="Emission_Gradient")
        mat_emission.use_nodes = True

        nodes = mat_emission.node_tree.nodes
        links = mat_emission.node_tree.links

        # Create nodes
        emission = nodes.new('ShaderNodeEmission')
        layer_weight = nodes.new('ShaderNodeLayerWeight')
        color_ramp = nodes.new('ShaderNodeValToRGB')  # Color Ramp node
        output = nodes['Material Output']

        # Connect nodes
        links.new(layer_weight.outputs['Facing'], color_ramp.inputs['Fac'])  # Connect layer weight to color ramp
        links.new(color_ramp.outputs['Color'], emission.inputs['Color'])  # Connect color ramp to emission
        links.new(emission.outputs['Emission'], output.inputs['Surface'])
        # Color Ramp settings
# Shader 1
#        color_ramp.color_ramp.elements[0].color = (1, 1, 1, 1)  # white
#        color_ramp.color_ramp.elements[0].position = 0.055  # position of white color
#        color_ramp.color_ramp.elements[1].color = (0.572, 0.572, 0.572, 1)  # black
#        color_ramp.color_ramp.elements[1].position = 0.170  # position of black color
#        color_ramp.color_ramp.elements.new(0.305)  # grey2
#        color_ramp.color_ramp.elements.new(0.517)  # black
#        color_ramp.color_ramp.elements[2].color = (0.438, 0.438, 0.438, 1)  # grey2
#        color_ramp.color_ramp.elements[3].color = (0, 0, 0, 1)  # black
#        color_ramp.color_ramp.interpolation = 'B_SPLINE'  # Set interpolation to B-Spline
#        emission.inputs['Strength'].default_value = 5  # Set emission strength

# Shader 2
#        color_ramp.color_ramp.elements[0].color = (1, 1, 1, 1)  # white
#        color_ramp.color_ramp.elements[0].position = 0.141  # position of white color
#        color_ramp.color_ramp.elements[1].color = (0.102, 0.102, 0.102, 1)  # black
#        color_ramp.color_ramp.elements[1].position = 0.263  # position of black color
#        color_ramp.color_ramp.elements.new(0.457)  # grey2
#        color_ramp.color_ramp.elements.new(0.636)  # black
#        color_ramp.color_ramp.elements[2].color = (0.031, 0.031, 0.031, 1)  # grey2
#        color_ramp.color_ramp.elements[3].color = (0, 0, 0, 1)  # black
#        color_ramp.color_ramp.interpolation = 'B_SPLINE'  # Set interpolation to B-Spline
#        emission.inputs['Strength'].default_value = 5  # Set emission strength
# Shader 3
        color_ramp.color_ramp.elements[0].color = (1, 1, 1, 1)  # white
        color_ramp.color_ramp.elements[0].position = 0.460  # position of white color
        color_ramp.color_ramp.elements[1].color = (0.102, 0.102, 0.102, 1)  # black
        color_ramp.color_ramp.elements[1].position = 0.670  # position of black color
        color_ramp.color_ramp.elements.new(0.820)  # grey2
        color_ramp.color_ramp.elements[2].color = (0.031, 0.031, 0.031, 1)  # grey2
        color_ramp.color_ramp.interpolation = 'B_SPLINE'  # Set interpolation to B-Spline
        emission.inputs['Strength'].default_value = 3  # Set emission strength


        # Layer Weight settings
        layer_weight.inputs['Blend'].default_value = 0.5  # adjust this value to control the effect of the layer weight
        # Emission strength settings

        sphere_obj.data.materials.append(mat_emission)  # Apply material to sphere      

      # Create a cylinder with Diffuse shader
        bpy.ops.mesh.primitive_cylinder_add(radius=(sphere_radius * 2.0), depth=sphere_radius * 2.0, location=location_inner)
        inner_circle_obj = bpy.context.object

        # Rotate the cylinder to match the plane and point towards origin
        inner_circle_obj.rotation_euler = rotation_quat.to_euler()

        # Add a Diffuse shader to the inner circle
        mat_diffuse = bpy.data.materials.new(name="Diffuse_Material")
        mat_diffuse.use_nodes = True
        mat_diffuse.node_tree.nodes.clear()

        nodes = mat_diffuse.node_tree.nodes
        links = mat_diffuse.node_tree.links

        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        node_diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')

        node_diffuse.inputs['Color'].default_value = (0, 0, 0, 1)  # Black color

        links.new(node_diffuse.outputs['BSDF'], node_output.inputs['Surface'])

        inner_circle_obj.data.materials.append(mat_diffuse)
          

def pickle_data(rw_mode, path, data):
    import pickle
    import gzip
    try:
        if rw_mode == READ:
            with gzip.open(path, 'rb') as f:
                data = pickle.load(f)
            return data
        elif rw_mode == WRITE:
            with gzip.open(path, 'wb') as f:
                pickle.dump(data, f)
        else:
            print('not support mode')
    except:
        print('file r/w error')
        return ERROR

    
def save_png_files():
    json_data = OrderedDict()
    json_data['CAMERA_0'] = {'Vertex': [], 'Project': [], 'OpenCV': [], 'Obj_Util': [], 'points3D': []}
    json_data['CAMERA_1'] = {'Vertex': [], 'Project': [], 'OpenCV': [], 'Obj_Util': [], 'points3D': []}

    for cam_name in camera_names:
        if cam_name not in bpy.data.objects or bpy.data.objects[cam_name].type != 'CAMERA':
            print({'ERROR'}, f"No camera found with name {cam_name}")
            break;    
        try:
            print(f"\n\n{cam_name}")
            print('--------TEST 1-----------')        
            points3D = origin_led_data
            print('points3D\n', points3D)
            points3D = np.array(points3D, dtype=np.float64)
            cam = bpy.data.objects[cam_name]
            # 이제 active object를 원하는 object로 변경
            bpy.context.view_layer.objects.active = cam
            # 선택한 오브젝트를 선택 상태
            cam.select_set(True)
            active_obj = bpy.context.active_object
            print('active_obj', active_obj)

            scene = bpy.context.scene
            scene.camera = cam  # 현재 씬의 카메라로 설정

            
            width = scene.render.resolution_x
            height = scene.render.resolution_y
            
            print('width', width, 'height', height)
            
            location = cam.location
            rotation = cam.rotation_euler
            quat = cam.rotation_quaternion
            # XYZ 오일러 각도를 degree 단위로 변환
            rotation_degrees = tuple(degrees(angle) for angle in rotation)
            # 결과 출력
            print(f"{cam} 위치: ", location)
            print(f"{cam} XYZ 오일러 회전 (도): ", rotation_degrees)
            print(f"{cam} XYZ 쿼너티온: ", quat)

            # print projection M
            P = get_3x4_P_matrix_from_blender_OpenCV(cam)
            projectionMatrix = np.matrix(P)
            print('projection M', projectionMatrix)

            # print R|T
            _, rvec, tvec = get_3x4_RT_matrix_from_blender_OpenCV(cam)
            R, _ = cv2.Rodrigues(rvec)
            rvec = np.array(R.ravel())
            tvec = np.array(tvec)
            print('rvec', rvec)
            print('tvec', tvec)

            intrinsic, rotationMatrix, homogeneousTranslationVector = cv2.decomposeProjectionMatrix(
                projectionMatrix)[:3]
            camT = -cv2.convertPointsFromHomogeneous(homogeneousTranslationVector.T)
            camR = Rot.from_matrix(rotationMatrix)
            blender_tvec = camR.apply(camT.ravel())
            blender_rvec = camR.as_rotvec()
            blender_rvec = blender_rvec.reshape(-1, 1)
            blender_tvec = blender_tvec.reshape(-1, 1)

            blender_image_points, _ = cv2.projectPoints(points3D, blender_rvec, blender_tvec,
                                                        cameraMatrix=intrinsic,
                                                        distCoeffs=None)
            blender_image_points = blender_image_points.reshape(-1, 2)

            print("Projected 2D image points <Projection>")
            print(blender_image_points)
            json_data[cam_name]['OpenCV'] = np.array(blender_image_points).tolist()
            json_data[cam_name]['points3D'] = np.array(points3D).tolist()

            print('\n\n')
            print('--------TEST 2-----------')
            # Insert your camera name here
            P, K, RT = get_3x4_P_matrix_from_blender(cam)
            print("K")
            print(K)
            print("RT")
            print(RT)
            print("P")
            print(P)

            for i, point in enumerate(points3D):
                point_3D = Vector(point)  # Convert tuple to Vector
                p = P @ Vector((point_3D.x, point_3D.y, point_3D.z, 1))  # Append 1 to the vector for matrix multiplication
                p /= p[2]

                print(f"Projected point {i}")
                print(p[:2])
                json_data[cam_name]['Project'].append(p[:2])
                print("proj by object_utils")
                obj_coords = project_by_object_utils(cam, point_3D)
                json_data[cam_name]['Obj_Util'].append(obj_coords)
                print(obj_coords)        

            # save png file
            scene.render.image_settings.file_format = 'PNG'
            scene.render.filepath = base_file_path + f"{cam_name}" + '_blender_test_image'
            bpy.ops.render.render(write_still=1)
            
        except KeyError:
            print('camera not found', camera_name)   

    print('json_data\n', json_data)
    jfile = base_file_path + 'blender_test_image.json'
    rw_json_data(1, jfile, json_data)

def make_camera_path(path_type, **kwargs):
    if path_type == 'sine_wave':
        print('make sine_wave')
        # Curve 데이터를 생성합니다.
        curve_data = bpy.data.curves.new(path_type, type='CURVE')
        curve_data.dimensions = '3D'

        # NurbsPath를 만들기 위한 설정을 합니다.
        polyline = curve_data.splines.new('NURBS')

        # 원점으로부터 반지름이 0.3인 구 표면 위를 0, -0.3, 0 부터 0, 0.3, 0 까지 사인파 모양으로 나타냅니다.
        num_points = 500  # 해상도를 높입니다
        polyline.points.add(num_points - 1)  # 포인트의 개수를 설정합니다. 
        radius = 0.3
        amplitude = 0.3  # 펄스의 진폭
        frequency = 5   # 펄스의 주파수

        for i in range(num_points):
            # y는 -0.3에서 0.3까지 선형으로 변화합니다.
            y = -0.3 + i * 0.6 / (num_points - 1)
            
            # angle은 y에 대응하는 각도입니다.
            angle = 2 * math.pi * frequency * y
            
            # xz 평면 위에서 사인파를 그립니다.
            x = radius * math.cos(angle)
            z = amplitude * math.sin(angle)

            polyline.points[i].co = (x, y, z, 1)

        # NurbsPath가 순환형 경로인지 설정합니다. 
        polyline.use_cyclic_u = True

        # 이 Curve 데이터를 Object로 만듭니다.
        curve_object = bpy.data.objects.new(path_type, curve_data)

        # 생성된 Object를 씬에 추가합니다.
        scene = bpy.context.scene
        scene.collection.objects.link(curve_object)

        # Path Animation의 프레임 수를 설정합니다.
        curve_object.data.path_duration = 500        

        animate_path(curve_object, 0, 100, 400)

        # Path Animation 활성화
        curve_object.data.use_path = True

    elif path_type == 'circle_curve':
        print('make circle_curve')
        radius = kwargs.get('radius', 0.3)  # 반지름이 제공되지 않은 경우, 기본값은 0.3으로 설정합니다.

        # Curve 데이터를 생성합니다.
        curve_data = bpy.data.curves.new(path_type, type='CURVE')
        curve_data.dimensions = '3D'

        # NurbsPath를 만들기 위한 설정을 합니다.
        polyline = curve_data.splines.new('NURBS')

        # 원점을 중심으로 반지름이 0.3인 원을 그립니다.
        num_points = 250  # 해상도를 높입니다
        polyline.points.add(num_points - 1)  # 포인트의 개수를 설정합니다. 

        for i in range(num_points):
            # angle은 i에 대응하는 각도입니다.
            angle = 2 * math.pi * i / num_points
            
            # xy 평면 위에서 원을 그립니다.
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)  # 원점을 중심으로 설정합니다.
            z = 0

            polyline.points[i].co = (x, y, z, 1)

        # NurbsPath가 순환형 경로인지 설정합니다. 
        polyline.use_cyclic_u = True

        # 이 Curve 데이터를 Object로 만듭니다.
        curve_object = bpy.data.objects.new(path_type, curve_data)

        # 생성된 Object를 씬에 추가합니다.
        scene = bpy.context.scene
        scene.collection.objects.link(curve_object)

        # Path Animation의 프레임 수를 설정합니다.
        curve_object.data.path_duration = 500        

        animate_path(curve_object, 0, 100, 400)

        # Path Animation 활성화
        curve_object.data.use_path = True

    elif path_type == 'quadrant_segment_circle':
        print('make quadrant_segment_circle')
        quadrant = kwargs['quadrant']  # 사분면 번호
        start_angle_degrees = kwargs['start_angle_degrees']  # 시작 각도
        radius = kwargs.get('radius', 0.3)
        # Path 이름 생성
        path_name = f"quad_circle_curve_{quadrant}_{start_angle_degrees}"

        # Curve 데이터를 생성합니다.
        curve_data = bpy.data.curves.new(path_name, type='CURVE')
        curve_data.dimensions = '3D'

        # NurbsPath를 만들기 위한 설정을 합니다.
        polyline = curve_data.splines.new('NURBS')

        # 원점을 중심으로 반지름이 0.3인 원의 일부를 그립니다.
        num_points = 250  # 해상도를 높입니다
        polyline.points.add(num_points - 1)  # 포인트의 개수를 설정합니다. 

        # 사분면에 따라 시작 각도를 설정합니다.
        if quadrant == 1:
            quadrant_start_angle_degrees = 0
        elif quadrant == 2:
            quadrant_start_angle_degrees = 90
        elif quadrant == 3:
            quadrant_start_angle_degrees = 180
        elif quadrant == 4:
            quadrant_start_angle_degrees = 270
        else:
            raise ValueError("Quadrant should be between 1 and 4.")

        # 각도를 라디안으로 변환합니다.

        start_angle = quadrant_start_angle_degrees * math.pi / 180
        end_angle = start_angle + start_angle_degrees * math.pi / 180  # 입력된 내각만큼 반원을 그립니다.


        for i in range(num_points):
            # angle은 i에 대응하는 각도입니다.
            angle = start_angle + i * (end_angle - start_angle) / (num_points - 1)
            
            # xy 평면 위에서 원을 그립니다.
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = 0

            polyline.points[i].co = (x, y, z, 1)

        # NurbsPath가 순환형 경로인지 설정합니다. 
        polyline.use_cyclic_u = False  # 원의 일부만 그리므로 순환형 경로는 아닙니다.

        # 이 Curve 데이터를 Object로 만듭니다.
        curve_object = bpy.data.objects.new(path_name, curve_data)

        # 생성된 Object를 씬에 추가합니다.
        scene = bpy.context.scene
        scene.collection.objects.link(curve_object)

        # Path Animation의 프레임 수를 설정합니다.
        curve_object.data.path_duration = 500        

        animate_path(curve_object, 0, 100, 400)

        # Path Animation 활성화
        curve_object.data.use_path = True
    elif path_type == 'axis_circle':
        print('make axis_circle')
        radius = kwargs.get('radius', 0.3)  # 반지름이 제공되지 않은 경우, 기본값은 0.3으로 설정합니다.

        # Curve 데이터를 생성합니다.
        curve_data = bpy.data.curves.new(path_type, type='CURVE')
        curve_data.dimensions = '3D'

        # NurbsPath를 만들기 위한 설정을 합니다.
        polyline = curve_data.splines.new('NURBS')

        # 원점을 중심으로 반지름이 0.3인 원을 그립니다.
        num_points = 250  # 해상도를 높입니다
        polyline.points.add(num_points - 1)  # 포인트의 개수를 설정합니다. 

        for i in range(num_points):
            # angle은 i에 대응하는 각도입니다.
            angle = 2 * math.pi * i / num_points
            
            # yz 평면 위에서 원을 그립니다.
            y = radius * math.cos(angle)
            z = radius * math.sin(angle)  # 원점을 중심으로 설정합니다.
            x = -0.3  # x축 위치를 -0.3으로 설정합니다.

            polyline.points[i].co = (x, y, z, 1)

        # NurbsPath가 순환형 경로인지 설정합니다. 
        polyline.use_cyclic_u = True

        # 이 Curve 데이터를 Object로 만듭니다.
        curve_object = bpy.data.objects.new(path_type, curve_data)

        # 생성된 Object를 씬에 추가합니다.
        scene = bpy.context.scene
        scene.collection.objects.link(curve_object)

        # Path Animation의 프레임 수를 설정합니다.
        curve_object.data.path_duration = 100        

#        animate_path(curve_object, 0, 100, 400)

        # Path Animation 활성화
        curve_object.data.use_path = True

        # Matrix.Rotation 메소드를 사용해서 회전 행렬을 생성합니다.
        # 첫 번째 인자는 회전 각도(여기서는 45도를 라디안으로 변환한 값), 두 번째 인자는 회전의 축을 나타내는 벡터입니다.
        rotation_matrix_3x3 = mathutils.Matrix.Rotation(math.radians(30), 3, 'Z')
        
        # 3x3 회전 행렬을 4x4로 변환합니다.
        rotation_matrix_4x4 = rotation_matrix_3x3.to_4x4()

        # Object에 회전 행렬을 적용합니다.
        curve_object.matrix_world = rotation_matrix_4x4
    elif path_type == 'robot_circle':
        print('make robot_circle')
        radius = kwargs.get('radius', 0.3)  # 반지름이 제공되지 않은 경우, 기본값은 0.3으로 설정합니다.
        slope = kwargs.get('slope', 0.0)  # 기울기가 제공되지 않은 경우, 기본값은 0으로 설정합니다. (이 경우 degree 단위)

        # Curve 데이터를 생성합니다.
        curve_data = bpy.data.curves.new(path_type, type='CURVE')
        curve_data.dimensions = '3D'

        # NurbsPath를 만들기 위한 설정을 합니다.
        polyline = curve_data.splines.new('NURBS')

        # 원점을 중심으로 반지름이 0.3인 원을 그립니다.
        num_points = 250  # 해상도를 높입니다
        polyline.points.add(num_points - 1)  # 포인트의 개수를 설정합니다. 

        for i in range(num_points):
            # angle은 i에 대응하는 각도입니다. pi/2를 빼서 시작점을 -y축 방향으로 이동시킵니다.
            # 여기서 각도 계산을 변경하여 원이 반대 방향으로 그려지게 했습니다.
            angle = 2 * math.pi * (num_points - i) / num_points - math.pi / 2

            # xy 평면 위에서 원을 그립니다.
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = 0  # z축 위치를 0으로 설정합니다.

            polyline.points[i].co = (x, y, z, 1)

        # NurbsPath가 순환형 경로인지 설정합니다. 
        polyline.use_cyclic_u = True
        # 이 Curve 데이터를 Object로 만듭니다.
        curve_object = bpy.data.objects.new(path_type, curve_data)
        # 생성된 Object를 씬에 추가합니다.
        scene = bpy.context.scene
        scene.collection.objects.link(curve_object)
        # Path Animation의 프레임 수를 설정합니다.
        curve_object.data.path_duration = 100      
        # Path Animation 활성화
        curve_object.data.use_path = True
        # Matrix.Rotation 메소드를 사용해서 회전 행렬을 생성합니다.
        # 첫 번째 인자는 회전 각도(여기서는 slope값을 그대로 사용), 두 번째 인자는 회전의 축을 나타내는 벡터입니다.
        rotation_matrix_3x3 = mathutils.Matrix.Rotation(math.radians(slope), 3, 'X')
        # 3x3 회전 행렬을 4x4로 변환합니다.
        rotation_matrix_4x4 = rotation_matrix_3x3.to_4x4()
        # Object에 회전 행렬을 적용합니다.
        curve_object.matrix_world = rotation_matrix_4x4
    elif path_type == 'custom_circle':
        print('make custom_circle')

        cam_location = kwargs.get('cam_location')
        cam_orientation = kwargs.get('cam_orientation')
        radius = kwargs.get('radius', math.sqrt(cam_location.x**2 + cam_location.y**2 + cam_location.z**2))

        # Get camera rotation values (pitch, yaw, roll)
        pitch = cam_orientation.x
        yaw = cam_orientation.y
        roll = cam_orientation.z

        curve_data = bpy.data.curves.new(path_type, type='CURVE')
        curve_data.dimensions = '3D'

        polyline = curve_data.splines.new('NURBS')

        num_points = 250
        polyline.points.add(num_points - 1)

        for i in range(num_points):
            angle = 2 * math.pi * (i + 0.5) / num_points 
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = 0  # keeping Z constant as we want the path to be in XY plane

            polyline.points[i].co = (x, y, z, 1)

        polyline.use_cyclic_u = True
        curve_object = bpy.data.objects.new(path_type, curve_data)
        scene = bpy.context.scene
        scene.collection.objects.link(curve_object)
        curve_object.data.path_duration = 100
        curve_object.data.use_path = True

        # Create rotation matrix using the full camera rotation (pitch, yaw, roll)
        rotation_matrix = mathutils.Euler((pitch, yaw, roll), 'XYZ').to_matrix()
        rotation_matrix_4x4 = rotation_matrix.to_4x4()

        # Convert camera's local rotation to global rotation
        curve_object.matrix_world = bpy.context.object.matrix_world @ rotation_matrix_4x4



def animate_path(curve_object, frame_start, frame_mid, frame_end):
    # Animation data를 만들고 필요하다면 초기화합니다.
    if curve_object.animation_data is None:
        curve_object.animation_data_create()
    
    # Action을 만들고 필요하다면 초기화합니다.
    if curve_object.animation_data.action is None:
        curve_object.animation_data.action = bpy.data.actions.new(name="PathAnimation")
    
    # Curve의 eval_time 속성에 대한 fcurve를 가져오거나 만듭니다.
    fcurves = curve_object.animation_data.action.fcurves
    eval_time_fcurve = next((fcurve for fcurve in fcurves if fcurve.data_path == "eval_time"), None)
    if eval_time_fcurve is None:
        eval_time_fcurve = fcurves.new(data_path="eval_time")
    
    # eval_time을 애니메이션화합니다.
    eval_time_fcurve.keyframe_points.add(4)  # 4개의 keyframe을 추가합니다.
    eval_time_fcurve.keyframe_points[0].co = (frame_start, 0.0)  # 시작 keyframe
    eval_time_fcurve.keyframe_points[0].interpolation = 'BEZIER'  # bezier interpolation

    eval_time_fcurve.keyframe_points[1].co = (frame_mid, 0.5)  # 중간 keyframe
    eval_time_fcurve.keyframe_points[1].interpolation = 'LINEAR'  # linear interpolation

    eval_time_fcurve.keyframe_points[2].co = (frame_end, 1.0)  # 끝 keyframe
    eval_time_fcurve.keyframe_points[2].interpolation = 'BEZIER'  # bezier interpolation

    eval_time_fcurve.keyframe_points[3].co = (500, 1.0)  # 최종 keyframe
    eval_time_fcurve.keyframe_points[3].interpolation = 'BEZIER'  # bezier interpolation


#def dist_coeffs():
    #import numpy as np
    #dist_coeffs = np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)
    #import bpy
    ## Define the distortion coefficients
    #k1, k2, p1, p2 = dist_coeffs.flatten()
    ## Enable use_nodes in the compositor
    #bpy.context.scene.use_nodes = True

    ## Get reference to the node tree
    #tree = bpy.context.scene.node_tree

    ## Clear all nodes
    #tree.nodes.clear()

    ## Create the Render Layers node
    #render_layers_node = tree.nodes.new(type='CompositorNodeRLayers')

    ## Create the Composite node
    #composite_node = tree.nodes.new(type='CompositorNodeComposite')

    ## Create the Movie Distortion node
    #movie_distortion_node = tree.nodes.new(type='CompositorNodeMovieDistortion')

    ## Set the distortion parameters
    #movie_distortion_node.distort = 1  # 1 means distortion, -1 means undistortion
    #movie_distortion_node.distort_method = 'POLYNOMIAL'
    #movie_distortion_node.k1 = k1
    #movie_distortion_node.k2 = k2
    #movie_distortion_node.k3 = p1
    #movie_distortion_node.k4 = p2

    ## Link the nodes
    #links = tree.links
    #links.new(render_layers_node.outputs[0], movie_distortion_node.inputs[0])
    #links.new(movie_distortion_node.outputs[0], composite_node.inputs[0])


def make_models(shape):        
    if shape == 'real':
        try:
            led_data = origin_led_data
            create_filled_emission_sphere(led_data, origin_led_dir, led_size)

            # Draw Direction
            for i in range(len(led_data)):
                start = tuple(led_data[i])
                end = tuple(led_data[i] + origin_led_dir[i])
                draw_line(start, end, (1, 0, 0, 1), f"line_{i+len(origin_led_dir)}")        
            
            # Create a parent object (an empty object)
            bpy.ops.object.empty_add(location=(0, 0, 0))
            parent_obj = bpy.context.object
            parent_obj.name = "Controller"

            # Find all objects whose name contains "Cylinder", "Line", or "Sphere"
            for obj in bpy.data.objects:
                if "Cylinder" in obj.name or "line" in obj.name or "Sphere" in obj.name:
                    obj.parent = parent_obj    
        
                
        except:
            print('exception')
    else:
        model_data = pickle_data(READ, model_pickle_file, None)
        led_data = model_data['LED_INFO']
        model_data = model_data['MODEL_INFO']

        for i, leds in enumerate(led_data):
            print(f"{i}, led: {leds}")  
        # Simulator
        #########################
        # numpy 배열로부터 좌표 데이터를 가져옵니다.
        model_data = np.array(model_data)
        # led_data를 numpy 배열로 변환합니다.
        led_data = np.array(led_data)
        # 이 함수 호출로 메시 객체를 생성하고 표면을 그립니다.
        create_mesh_object(model_data, name=MESH_OBJ_NAME, padding=padding)

        # # 모델 오브젝트를 찾습니다.
        model_obj = bpy.data.objects[MESH_OBJ_NAME]
        led_objects = create_circle_leds_on_surface(led_data, led_size, shape)

        # 여기서 오브젝트 깎음
        for led_obj in led_objects:
            apply_boolean_modifier(model_obj, led_obj)
        #########################


def make_cameras_model():

    real_camera_data = pickle_data(READ, camera_pickle_file, None)            
                
#            # Use Real Camera R|T from Python Simulator
#            LRVEC = real_camera_data['L_CAM']['rvec'][0]
#            LTVEC = real_camera_data['L_CAM']['tvec'][0]
#            RRVEC = real_camera_data['R_CAM']['rvec'][0]
#            RTVEC = real_camera_data['R_CAM']['tvec'][0]

#            print('LRVEC\n', LRVEC)
#            print('LTVEC\n', LTVEC)
#            print('RRVEC\n', RRVEC)
#            print('RTVEC\n', RTVEC)

#            make_cameras("CAMERA_0", LRVEC, LTVEC, cam_0_matrix)
#            make_cameras("CAMERA_1", RRVEC, RTVEC, cam_1_matrix)

#            #Custom Camera Posion
#            rvec_left = np.array([ 1.2091998,   1.20919946, -1.20919946])
#            tvec_left = np.array([-2.82086248e-08, -2.35607960e-08,  2.00000048e-01])

#            rvec_right = np.array([ 0.86044094,  1.63833515, -1.63833502])
#            tvec_right = np.array([-7.45058060e-08, -1.88919191e-08,  2.00000152e-01])

    LRVEC = np.array([ 0.865009,   -1.37796869,  0.7863227 ])
    LTVEC = np.array([ 0.02188013, -0.0133368,   0.32024709])
    make_cameras("CAMERA_0", LRVEC, LTVEC, cam_0_matrix)
 
    # Default
    #default_rvec_left = np.array([1.5707964, 0.0, 0.0])
    #default_tvec_left = np.array([0.0, 0.0, 0.4])

    default_rvec_left = np.array([0.0, 0.0, 0.0])
    default_tvec_left = np.array([0.0, 0.0, 0.0])

#    default_rvec_right = np.array([0.0, 0.0, 0.0])
#    default_tvec_right = np.array([0.0, 0.0, 0.0])
    make_cameras("CAMERA_0_DEFAULT", default_rvec_left, default_tvec_left, cam_0_matrix)
#        make_cameras("CAMERA_1", default_rvec_right, default_tvec_right, cam_1_matrix)
#    make_cameras_default("CAMERA_0", rvec_left, tvec_left)
#    make_cameras_default("CAMERA_1", rvec_right, tvec_right)


def draw_camera_pos_and_dir():  

    ## DRAW Camera pos and dir
    deviation_data = pickle_data(READ, camera_info_path, None)
    scale_factor = 0.5  # Scale factor for adjusting quiver length

    # Create a new material
    mat = bpy.data.materials.new(name="RedMaterial")
    mat.diffuse_color = (1.0, 0.0, 0.0, 1.0)  # Red color

    for key, camera_info in deviation_data['CAMERA_INFO'].items():
        if 'BL' in key:
            cam_id = int(key.split('_')[1])
            if cam_id == 1:
                continue
            
            rvec = camera_info['rt']['rvec']
            tvec = camera_info['rt']['tvec']
            
            # Add Camera Position
            position, rotation_quat = blender_location_rotation_from_opencv(rvec, tvec)        
            # Convert quaternion rotation to Euler rotation
            rotation_mat = rotation_quat.to_matrix().to_4x4()
            rotation_euler = rotation_mat.to_euler()
            
            # Create a small sphere at the camera position
            bpy.ops.mesh.primitive_uv_sphere_add(location=position, radius=0.0005)
            sphere = bpy.context.object
            sphere.data.materials.append(mat)  # Assign the material to the sphere 

            # Adjust start position of the cylinderl
            start_position = position + rotation_quat @ Vector((0, 0, 0.0005))        
            # Create quiver
            bpy.ops.mesh.primitive_cylinder_add(radius=0.0001, depth=np.linalg.norm(rotation_euler)*scale_factor,
                                             location=start_position, rotation=rotation_euler)
            
            points3d = camera_info['remake_3d']
            for blobs in points3d:
                bpy.ops.mesh.primitive_uv_sphere_add(location=blobs, radius=0.001)
                blob_sphere = bpy.context.object
                blob_sphere.data.materials.append(mat)  # Assign the material to the sphere 

            print('position', position)
            print('rotation', rotation_euler)          
            

    # Create a blue material
    blue_mat = bpy.data.materials.new(name="BlueMaterial")
    blue_mat.diffuse_color = (0, 0, 1, 1)  # R, G, B, alpha

    ba_data = pickle_data(READ, ba_result_path, None)

    ba_l_cam_param = ba_data['l_cam_param']
    ba_r_cam_param = ba_data['r_cam_param']

    for camera_rt in ba_l_cam_param:
        rvec = camera_rt[:3]
        tvec = camera_rt[3:]
        
        # Add Camera Position
        position, rotation_quat = blender_location_rotation_from_opencv(rvec, tvec)        
        # Convert quaternion rotation to Euler rotation
        rotation_mat = rotation_quat.to_matrix().to_4x4()
        rotation_euler = rotation_mat.to_euler()

        # Create a small sphere at the camera position
        bpy.ops.mesh.primitive_uv_sphere_add(location=position, radius=0.0005)
        sphere = bpy.context.object
        sphere.data.materials.append(blue_mat)  # Assign the blue material to the sphere 

        # Adjust start position of the cylinder
        start_position = position + rotation_quat @ Vector((0, 0, 0.0005))        
        # Create quiver
        bpy.ops.mesh.primitive_cylinder_add(radius=0.0001, depth=np.linalg.norm(rotation_euler)*scale_factor,
                                             location=start_position, rotation=rotation_euler)
        cylinder = bpy.context.object
        cylinder.data.materials.append(blue_mat)  # Assign the blue material to the cylinder


def log_and_render(cam_name, frame, f, render_folder, f_transform):
    cam = bpy.data.objects[cam_name]
    bpy.context.view_layer.objects.active = cam
    cam.select_set(True)

    scene = bpy.context.scene
    scene.camera = cam

    _, rvec, tvec = get_3x4_RT_matrix_from_blender_OpenCV(cam)
    R, _ = cv2.Rodrigues(rvec)
    rvec = np.array(R.ravel())
    tvec = np.array(tvec)
    f.write(f"Frame:{frame}, Rvec:{rvec}, Tvec:{tvec}\n")        
    
    # 카메라의 위치와 회전을 4x4 변환 행렬로 변환
    position, rotation_quat = blender_location_rotation_from_opencv(rvec, tvec)        
    # 회전 정보를 4x4 행렬로 변환
    rotation_mat = rotation_quat.to_matrix()
    # 4x4 변환 행렬 생성
    T_camera = np.eye(4)
    T_camera[:3, :3] = np.array(rotation_mat)
    T_camera[:3, 3] = np.array(position)

    # 변환 행렬의 역행렬을 계산
    T_inverse = np.linalg.inv(T_camera)
    # Convert the matrix to a 1D list
    T_inverse_flat = T_inverse.ravel()
    # Convert the numpy array to a comma-separated string
    T_inverse_str = ' '.join(str(x) for x in T_inverse_flat)
    f_transform.write(f"Frame:{frame}, T_inverse:[{T_inverse_str}]\n") 
    scene.render.filepath = os.path.join(render_folder, f"frame_{frame:04d}.png")
    bpy.ops.render.render(write_still=True)


def render_camera_pos_and_png(cam_name, **kwargs):
    start_frame = kwargs.get('start_frame')
    end_frame = kwargs.get('end_frame')
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'

    os.makedirs(render_folder, exist_ok=True)

    camera_log_path = os.path.join(render_folder, "camera_log.txt")
    transformations_log_path = os.path.join(render_folder, "transformations.txt")


    with open(camera_log_path, 'w') as f:
        f.write("")
    
    with open(transformations_log_path, 'w') as f_transform:
        f_transform.write("")


    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)

        with open(camera_log_path, 'a') as f, open(transformations_log_path, 'a') as f_transform:
            log_and_render(cam_name, frame - start_frame + 1, f, render_folder, f_transform)


def render_image_inverse(cam_name):
    inv_rvec_left = np.array([3.14159265, 0.0, 0.0])
    inv_tvec_left = np.array([0.0, 0.0, 0.0])
    make_cameras(cam_name, inv_rvec_left, inv_tvec_left, cam_0_matrix)
    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'

    transformations_log_path = os.path.join(render_folder, "transformations.txt")

    with open(transformations_log_path, 'r') as file:
        lines = file.readlines()

    # print(lines)
    inverse_rt = {}
    for line in lines:
        parts = line.split(',')
        frame = int(parts[0].split(':')[1].strip())
        T_INV = list(map(float, parts[1].split(':')[1].strip()[1:-1].split()))
        inverse_rt[frame] = T_INV
    
#    for frame, data in inverse_rt.items():
#        print(f"{frame}\n{data}")
        
    os.makedirs(render_folder + 'inv/', exist_ok=True)
    # Assuming there's a specific object you want to transform
    obj = bpy.data.objects["Controller"]  # We are using the name of our grouped objects

    # Fetch the transformations from the dictionary
    for frame, transformation in inverse_rt.items():
        scene.frame_set(frame)

        # Convert the 1D list back to a 4x4 matrix
        transformation = np.array(transformation).reshape(4, 4)

        # Convert the numpy array to a Blender Matrix
        transformation = Matrix(transformation.tolist())

        # Apply the transformation to the object
        obj.matrix_world = transformation
        
        # Set the active camera for rendering
        camera = bpy.data.objects[cam_name]
        scene.camera = camera

        # Set the output path and render
        scene.render.filepath = os.path.join(render_folder + 'inv/', f"frame_inv_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)




# Make camera follow path
def make_camera_follow_path(camera, path, **kwargs):
    start_frame = kwargs.get('start_frame', -1)
    end_frame = kwargs.get('end_frame', -1)
    follow_path = camera.constraints.new(type='FOLLOW_PATH')
    follow_path.target = path
    follow_path.use_curve_follow = True
    
    if start_frame > -1 and end_frame > -1:
        # Create a keyframe for the follow_path offset_factor at the start frame
        follow_path.offset_factor = 1  # Set offset_factor to 1 at the start frame
        follow_path.keyframe_insert(data_path="offset_factor", frame=start_frame)

        # Create a keyframe for the follow_path offset_factor at the end frame
        follow_path.offset_factor = 0  # Set offset_factor to 0 at the end frame
        follow_path.keyframe_insert(data_path="offset_factor", frame=end_frame)


# Make camera look at an object
def make_camera_look_at(camera, obj):
    # Remove previous tracking constraints
    for constraint in camera.constraints:
        if constraint.type in ['DAMPED_TRACK', 'LOCKED_TRACK', 'TRACK_TO']:
            camera.constraints.remove(constraint)            
    # Add track to constraint
    track_to = camera.constraints.new(type='TRACK_TO')
    track_to.target = obj
    track_to.track_axis = 'TRACK_NEGATIVE_Z'
    track_to.up_axis = 'UP_Z'


# Make camera track to path normal
def make_camera_track_to_path_normal(camera, path):
    for point in path.data.splines[0].bezier_points:
        # Calculate path normal from tilt
        normal = np.array([math.sin(point.tilt), 0, math.cos(point.tilt)])        
        # Apply path normal to camera rotation
        camera.rotation_euler = normal

def make_camera_track_to(camera, target):
    # Create track to constraint and configure it
    track_to = camera.constraints.new(type='TRACK_TO')
    track_to.target = target
    track_to.track_axis = 'TRACK_NEGATIVE_Z'
    track_to.up_axis = 'UP_Y'
    
# Set camera roll
def set_camera_roll(camera, x_angle, y_angle, z_angle):
    # Angle should be in radians
    camera.rotation_euler.x = x_angle
    camera.rotation_euler.y = y_angle
    camera.rotation_euler.z = z_angle


def draw_camera_recording(cam_name, default_cam_name):        
    # Get the camera object
    camera = bpy.data.objects[cam_name]
    default_camera = bpy.data.objects[default_cam_name]
    # Create an empty object at the same Z position as the camera
    bpy.ops.object.empty_add(location=(0, 0, camera.location.z))
    empty_obj = bpy.context.object
    empty_obj.name = f"EMPTY_{cam_name}"

    #make_camera_path('sine_wave')
    #make_camera_path('circle_curve', radius=0.3)
    #make_camera_path('quadrant_segment_circle', quadrant=3, start_angle_degrees=45, radius=0.3)
    #make_camera_path('quadrant_segment_circle', quadrant=2, start_angle_degrees=45, radius=0.3)
    #make_camera_path('axis_circle', radius=0.15)
    #make_camera_path('robot_circle', radius=0.4, slope=30.0)
    #make_camera_path('custom_circle', cam_location=camera.location, cam_orientation=camera.rotation_euler)

    # Add driver to change Z position of empty object as the camera moves
#    driver = empty_obj.driver_add("location", 2)  # 2 for Z axis
#    var = driver.driver.variables.new()
#    var.name = "z"
#    var.type = 'TRANSFORMS'
#    var.targets[0].id = camera
#    var.targets[0].transform_type = 'LOC_Z'
#    driver.driver.expression = "z"



def custom_camera_tracker(cam_name, default_cam_name):
    path_name = f"{cam_name}_CIRCLE_LOOP"
    camera = bpy.data.objects[cam_name]
    default_camera = bpy.data.objects[default_cam_name]
    cam_location = camera.location
    cam_orientation = camera.rotation_euler
    radius = math.sqrt(cam_location.x**2 + cam_location.y**2 + cam_location.z**2)

    # Get camera rotation values (pitch, yaw, roll)
    pitch = cam_orientation.x - math.radians(90)  # subtracting 90 degrees
    yaw = cam_orientation.y
    roll = cam_orientation.z

    curve_data = bpy.data.curves.new(path_name, type='CURVE')
    curve_data.dimensions = '3D'

    polyline = curve_data.splines.new('NURBS')

    num_points = 360
    polyline.points.add(num_points - 1)
    for i in range(num_points):
        angle = 2 * math.pi * (num_points - i) / num_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 0  # keeping Z constant as we want the path to be in XY plane
        polyline.points[i].co = (x, y, z, 1)

    polyline.use_cyclic_u = True
    curve_object = bpy.data.objects.new(path_name, curve_data)
    scene = bpy.context.scene
    scene.collection.objects.link(curve_object)
    curve_object.data.path_duration = 360
    curve_object.data.use_path = True

    # Create rotation matrix using the full camera rotation (pitch, yaw, roll)
    rotation_matrix = mathutils.Euler((pitch, yaw, roll), 'XYZ').to_matrix()
    rotation_matrix_4x4 = rotation_matrix.to_4x4()

    # Convert camera's local rotation to global rotation
    curve_object.matrix_world = bpy.context.object.matrix_world @ rotation_matrix_4x4
    # Convert Euler's rotation to Quaternion
    rotation_quat = cam_orientation.to_quaternion()

    direction = rotation_quat @ mathutils.Vector((0.0, 0.0, -1.0))
    curve_object.location = camera.location + radius * direction.normalized()        

    polyline_points_world = [curve_object.matrix_world @ p.co for p in polyline.points]
    # Calculate the difference vector and extend it to 4D
    diff_vector = mathutils.Vector((curve_object.location.x, curve_object.location.y, curve_object.location.z, 0))
    # Apply the difference to each point
    polyline_points_world = [p + diff_vector for p in polyline_points_world]
#    print('polyline_points_world\n',polyline_points_world)
#    # After defining and populating polyline_points_world
#    for i, p in enumerate(polyline_points_world):
#        bpy.ops.mesh.primitive_uv_sphere_add(location=p.xyz, radius=0.001)
#        sphere = bpy.context.object
#        sphere.name = f"DEBUG_SPHERE_{i}"
#        sphere.data.materials.append(bpy.data.materials.new(name="Red"))
#        sphere.data.materials[0].diffuse_color = (1, 0, 0, 1)  # Set to red color


    # Calculate the distances from the camera to each point
    distances = [(cam_location - p.xyz).length for p in polyline_points_world]

    # Find the index of the closest point
    closest_point_index = distances.index(min(distances))
    print('closest_point_index ', closest_point_index)

    # Create an empty object at the center of the path
    bpy.ops.object.empty_add(location=curve_object.location)
    center_target = bpy.context.object
    center_target.name = f'{path_name}_CENTER_TARGET'  # Name the empty object
    # Create a red sphere at the center_target location
    bpy.ops.mesh.primitive_uv_sphere_add(location=center_target.location, radius=0.005)
    sphere = bpy.context.object
    sphere.name = f"CENTER_POINT"
    mat = bpy.data.materials.new(name="Center")
    mat.diffuse_color = (1.0, 0, 0, 1.0)  # Set to red color
    sphere.data.materials.append(mat)

    # Set the animation start and end frames
    start_frame = closest_point_index
    end_frame = closest_point_index + 360

    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame    
    
    # Create a red sphere at the closest point for debugging
    bpy.ops.mesh.primitive_uv_sphere_add(location=polyline_points_world[closest_point_index].xyz, radius=0.005)
    sphere = bpy.context.object
    sphere.name = f"START_POINT_{start_frame}"
    mat = bpy.data.materials.new(name="START")
    mat.diffuse_color = (1.0, 0, 0, 1.0)
    sphere.data.materials.append(mat)


    # At the beginning of your function
    relative_path_collection = bpy.data.collections.new('RELATIVE_PATH')
    bpy.context.scene.collection.children.link(relative_path_collection)

    relative_path_collection.objects.link(bpy.data.objects.get('CAMERA_0_CIRCLE_LOOP'))
    relative_path_collection.objects.link(bpy.data.objects.get('CAMERA_0_CIRCLE_LOOP_CENTER_TARGET'))
    relative_path_collection.objects.link(bpy.data.objects.get('CENTER_POINT'))
    relative_path_collection.objects.link(bpy.data.objects.get('START_POINT_90'))

