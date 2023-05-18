import bpy
from mathutils import Matrix, Vector
import bpy_extras
from bpy_extras.object_utils import world_to_camera_view
from math import degrees
from datetime import datetime
import platform
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot
import math
import json
from collections import OrderedDict

os_name = platform.system()
base_file_path = ''
if os_name == 'Windows':
    print("This is Windows")
    base_file_path = 'D:/OpenCV_APP/led_pos_simulation/blender_3d/image_output/'
elif os_name == 'Linux':
    print("This is Linux")
    base_file_path = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/blender_3d/image_output/'
else:
    print("Unknown OS")


def delete_all_objects_except(exclude_object_names):
    for obj in bpy.data.objects:
        if obj.name not in exclude_object_names:
            bpy.data.objects.remove(obj, do_unlink=True)


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


def make_cameras(camera_name, rvec, tvec):
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


image_file_path = base_file_path
blend_file_path = base_file_path + 'blender_test_image.blend'

exclude_object_names = ["Cube", "CAMERA_0", "CAMERA_1"]
delete_all_objects_except(exclude_object_names)

# rvec_left = np.array([ 1.2091998,   1.20919946, -1.20919946])
# tvec_left = np.array([-2.82086248e-08, -2.35607960e-08,  2.00000048e-01])

# rvec_right = np.array([ 0.86044094,  1.63833515, -1.63833502])
# tvec_right = np.array([-7.45058060e-08, -1.88919191e-08,  2.00000152e-01])

# make_cameras("CAMERA_0", rvec_left, tvec_left)
# make_cameras("CAMERA_1", rvec_right, tvec_right)


data_objects = bpy.data.objects  # 사용자 입력으로 카메라 이름 가져오기

obj_camera = bpy.context.scene.camera.name
print('obj_camera', obj_camera)
scene = bpy.context.scene

scene.render.resolution_x = 1280
scene.render.resolution_y = 960
width = scene.render.resolution_x
height = scene.render.resolution_y

json_data = OrderedDict()
json_data['CAMERA_0'] = {'Vertex': [], 'Project': [], 'OpenCV': [], 'Obj_Util': [], 'points3D': []}
json_data['CAMERA_1'] = {'Vertex': [], 'Project': [], 'OpenCV': [], 'Obj_Util': [], 'points3D': []}

for obj in data_objects:
    if 'CAMERA' in obj.name:
        camera_name = obj.name

        cam = bpy.data.objects[camera_name]
        obj = bpy.data.objects['Cube']

        limit = 0.1
        DeselectEdgesAndPolygons(obj)

        mWorld = obj.matrix_world
        vertices = [mWorld @ v.co for v in obj.data.vertices]

        # Create a new material with an emission shader
        led_material = bpy.data.materials.new(name="Emission")
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

        print(f"\n{camera_name}\n")
        print('--------TEST 1-----------')
        points3D = []

        for i, v in enumerate(vertices):
            co2D = world_to_camera_view(scene, cam, v)

            # Add a small sphere at each vertex position
            bpy.ops.mesh.primitive_uv_sphere_add(location=v)
            bpy.ops.transform.resize(value=(0.01, 0.01, 0.01))
            sphereObject = bpy.context.object

            # Assign the emission material to the sphere
            if sphereObject.data.materials:
                sphereObject.data.materials[0] = led_material
            else:
                sphereObject.data.materials.append(led_material)

            if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0 and co2D.z > 0:
                depsgraph = bpy.context.evaluated_depsgraph_get()
                location = scene.ray_cast(depsgraph, cam.location, (v - cam.location).normalized())
                if location[0] and (v - location[1]).length < limit:
                    print(f"Vertex {i}: 3D coord: {v}, 2D coord: {(co2D.x * width, co2D.y * height)}")
                    bpy.ops.object.text_add(location=v)
                    text_obj = bpy.context.object
                    text_obj.data.body = f"V{i}"
                    text_obj.scale *= 0.1

                    # Append the 3D coordinates to the points3D list as a tuple
                    points3D.append(tuple(v))

            bpy.ops.object.duplicate()  # Keep the sphere object in the scene for rendering

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

        # ProjectPoints
        print('points3D\n', points3D)
        points3D = np.array(points3D, dtype=np.float64)
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
        json_data[camera_name]['OpenCV'] = np.array(blender_image_points).tolist()
        json_data[camera_name]['points3D'] = np.array(points3D).tolist()

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
            json_data[camera_name]['Project'].append(p[:2])
            print("proj by object_utils")
            obj_coords = project_by_object_utils(cam, point_3D)
            json_data[camera_name]['Obj_Util'].append(obj_coords)
            print(obj_coords)

print('json_data\n', json_data)
jfile = image_file_path + 'blender_test_image.json'
rw_json_data(1, jfile, json_data)

# save png file
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = image_file_path + f"{obj_camera}" + '_blender_test_image.png'
bpy.ops.render.render(write_still=1)

# bpy.ops.wm.save_as_mainfile(filepath=blend_file_path)
