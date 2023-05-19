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


'''
Functions
'''


def delete_all_objects_except(exclude_object_names):
    for obj in bpy.data.objects:
        if obj.name not in exclude_object_names:
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
        emission_node.inputs['Strength'].default_value = 0.01  # 강도 조절
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
    print('R_BlenderView_to_OpenCVView', R_BlenderView_to_OpenCVView)
    location, rotation = obj.matrix_world.decompose()[:2]
    print('location', location, 'rotation', rotation)
    # Convert the rotation to axis-angle representation
    axis, angle = rotation.to_axis_angle()

    # Create a 3x3 rotation matrix from the axis-angle representation
    R_BlenderView = Matrix.Rotation(angle, 3, axis).transposed()

    T_BlenderView = -1.0 * R_BlenderView @ location

    R_OpenCV = R_BlenderView_to_OpenCVView @ R_BlenderView
    T_OpenCV = R_BlenderView_to_OpenCVView @ T_BlenderView

    R, _ = cv2.Rodrigues(R_OpenCV)
    print('R_OpenCV', R_OpenCV)
    print('R_OpenCV(Rod)', R.ravel())
    print('T_OpenCV', T_OpenCV)

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


def create_camera(camera_f, camera_c, cam_location, name, rot):
    # roll = math.radians(-rot[0])
    # pitch = math.radians(rot[1])
    # yaw = math.radians(90 + rot[2])
    rotation = quaternion_to_euler_degree(rot)
    print('euler(degree)', rotation)

    X, Y, Z = cam_location
    print('position', X, Y, Z)
    # Remove existing camera
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

    # MAKE DEFAULT CAM
    # bpy.ops.object.camera_add(location=(0.2, 0, 0))
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
    # MAKE DEFAULT CAM
    cam.rotation_euler = (math.radians(rotation[0]), math.radians(rotation[1]), math.radians(rotation[2]))
    # cam.rotation_euler = (yaw, roll, pitch)

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
    sensor_width_mm = 36.0  # Assuming 35mm camera sensor size
    sensor_height_mm = 27.0  # Assuming 35mm camera sensor size
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
    cam.data.lens_unit = 'FOV'
    cam.data.angle = 2 * math.atan(sensor_width / (2 * focal_length))  # Field of view in radians
    cam.data.sensor_width = sensor_width
    cam.data.sensor_height = sensor_height

    scene = bpy.context.scene

    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x

    view_fac_in_px = sensor_width_px

    # cam.data.shift_x = (cx - sensor_width_px / 2.0) / fx_px # Shift in X direction
    # cam.data.shift_y = (cy - sensor_height_px / 2.0) / fy_px # Shift in Y direction

    cam.data.shift_x = (sensor_width_px / 2 - cx_px) / view_fac_in_px
    cam.data.shift_y = (cy_px - sensor_height_px / 2) * pixel_aspect_ratio / view_fac_in_px

    print('shift_x, shift_y', cam.data.shift_x, cam.data.shift_y)
    return cam


def create_camera_default(cam_location, rot, name):
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



def draw_line(p1, p2, name):
    p1 = Vector(p1)
    p2 = Vector(p2)
    direction = p2 - p1
    scaled_direction = direction * 10
    p2 = p1 + scaled_direction

    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.0001, depth=scaled_direction.length, end_fill_type='NGON',
                                        location=(0, 0, 0), scale=(1, 1, 1))
    line = bpy.context.object
    line.name = name
    line.location = (p1 + p2) / 2
    line.rotation_mode = 'QUATERNION'
    line.rotation_quaternion = scaled_direction.to_track_quat('Z', 'Y')


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


'''
TEST START
'''
print('\n\n\n')
print('TEST START')

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

# shape = 'plane'
# shape = 'sphere'
# shape = 'cylinder'
# shape = 'cylinder_base'
shape = 'basic'

pickle_file = None
os_name = platform.system()
if os_name == 'Windows':
    print("This is Windows")
    if shape == 'sphere':
        pickle_file = 'D:/OpenCV_APP/led_pos_simulation/find_pos_legacy/result.pickle'
    elif shape == 'cylinder':
        pickle_file = 'D:/OpenCV_APP/led_pos_simulation/find_pos_legacy/result_cylinder.pickle'
    elif shape == 'cylinder_base':
        pickle_file = 'D:/OpenCV_APP/led_pos_simulation/find_pos_legacy/result_cylinder_base.pickle'
    else:
        pickle_file = 'D:/OpenCV_APP/led_pos_simulation/find_pos_legacy/basic_test.pickle'
        base_file_path = 'D:/OpenCV_APP/led_pos_simulation/blender_3d/image_output/blender_basic/'

elif os_name == 'Linux':
    print("This is Linux")
    if shape == 'sphere':
        pickle_file = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/find_pos_legacy/result.pickle'
    elif shape == 'cylinder':
        pickle_file = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/find_pos_legacy/result_cylinder.pickle'
    elif shape == 'cylinder_base':
        pickle_file = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/find_pos_legacy/result_cylinder_base.pickle'
    else:
        pickle_file = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/find_pos_legacy/basic_test.pickle'
        base_file_path = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/blender_3d/image_output/blender_basic/'
    
else:
    print("Unknown OS")

with gzip.open(pickle_file, 'rb') as f:
    data = pickle.load(f)

led_data = data['LED_INFO']
model_data = data['MODEL_INFO']
camera_names = ["CAMERA_0", "CAMERA_1"]



for i, leds in enumerate(led_data):
    print(f"{i}, led: {leds}")


padding = 0.0  # 원하는 패딩 값을 입력하세요.
# LED 원의 반지름을 설정합니다. 원하는 크기를 입력으로 제공할 수 있습니다.
led_size = 0.003
led_thickness = 0.001

default_cameraK = {'serial': "default", 'camera_f': [1, 1], 'camera_c': [0, 0]}
cam_0_matrix = {'serial': "WMTD306N100AXM", 'camera_f': [712.623, 712.623], 'camera_c': [653.448, 475.572]}
cam_1_matrix = {'serial': "WMTD305L6003D6", 'camera_f': [716.896, 716.896], 'camera_c': [668.902, 460.618]}


# delte objects
exclude_object_names = ["Oculus_L_05.002"]
delete_all_objects_except(exclude_object_names)

MESH_OBJ_NAME = 'MeshObject_' + f'{shape}'
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


'''
# Real Controller
#########################
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
#########################
'''

image_file_path = base_file_path
blend_file_path = base_file_path + 'blender_test_image.blend'

rvec_left = np.array([ 1.2091998,   1.20919946, -1.20919946])
tvec_left = np.array([-2.82086248e-08, -2.35607960e-08,  2.00000048e-01])

rvec_right = np.array([ 0.86044094,  1.63833515, -1.63833502])
tvec_right = np.array([-7.45058060e-08, -1.88919191e-08,  2.00000152e-01])


#make_cameras("CAMERA_0", rvec_left, tvec_left, cam_0_matrix)
#make_cameras("CAMERA_1", rvec_right, tvec_right, cam_0_matrix)
make_cameras_default("CAMERA_0", rvec_left, tvec_left)
make_cameras_default("CAMERA_1", rvec_right, tvec_right)


# 카메라 해상도 설정 (예: 1920x1080)
bpy.context.scene.render.resolution_x = 1280
bpy.context.scene.render.resolution_y = 960

# 렌더링 결과의 픽셀 밀도를 100%로 설정 (기본값은 50%)
bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.render.film_transparent = False  # 렌더링 배경을 불투명하게 설정

bpy.context.scene.render.engine = 'CYCLES'
# bpy.context.scene.cycles.transparent_max_bounces = 12  # 반사와 굴절 최대 반투명 경계 설정
# bpy.context.scene.cycles.preview_samples = 100  # 뷰포트 렌더링 품질 설정
bpy.context.scene.unit_settings.system = 'METRIC'
bpy.context.scene.unit_settings.scale_length = 1.0
bpy.context.scene.unit_settings.length_unit = 'METERS'

# 렌더링 엔진을 Eevee로 설정합니다.
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
# Eevee 렌더링 설정을 조절합니다.
#bpy.context.scene.eevee.use_bloom = True
#bpy.context.scene.eevee.bloom_threshold = 0.8
#bpy.context.scene.eevee.bloom_radius = 6.5
#bpy.context.scene.eevee.bloom_intensity = 0.1

# 월드 배경을 어둡게 설정합니다.
set_up_dark_world_background()




json_data = OrderedDict()
json_data['CAMERA_0'] = {'Vertex': [], 'Project': [], 'OpenCV': [], 'Obj_Util': [], 'points3D': []}
json_data['CAMERA_1'] = {'Vertex': [], 'Project': [], 'OpenCV': [], 'Obj_Util': [], 'points3D': []}

for cam_name in camera_names:
    if cam_name not in bpy.data.objects or bpy.data.objects[cam_name].type != 'CAMERA':
        print({'ERROR'}, f"No camera found with name {camera_name}")
        break;
    
    try:
        print(f"\n\n{cam_name}")
        print('--------TEST 1-----------')        
        points3D = led_data
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
        scene.render.filepath = base_file_path + f"{cam_name}" + '_blender_test_image.png'
        bpy.ops.render.render(write_still=1)
        
    except KeyError:
        print('camera not found', camera_name)   

print('json_data\n', json_data)
jfile = base_file_path + 'blender_test_image.json'
rw_json_data(1, jfile, json_data)



# List of camera names
set_track_to = 1
if set_track_to == 1:
    # List of camera names
    for camera_name in camera_names:
        # Get the camera object
        camera = bpy.data.objects[camera_name]

        # Create an empty object at the origin
        bpy.ops.object.add(type='EMPTY', location=(0, 0, 0))
        empty_obj = bpy.context.active_object
        empty_obj.name = f"EMPTY_{camera_name}"

        # Make the empty object the parent of the camera
        camera.parent = empty_obj

        # Add a 'Track To' constraint to the camera to keep it pointed at the origin
        constraint = camera.constraints.new(type='TRACK_TO')
        constraint.target = empty_obj
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
