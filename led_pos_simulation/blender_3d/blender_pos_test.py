import bpy
from mathutils import Matrix, Vector
from bpy_extras.object_utils import world_to_camera_view
from math import degrees
from datetime import datetime
import platform
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot

os_name = platform.system()
base_file_path = ''
if os_name == 'Windows':
    print("This is Windows")
    base_file_path = 'D:/OpenCV_APP/led_pos_simulation/blender_3d/blender_test_image'
elif os_name == 'Linux':
    print("This is Linux")
    base_file_path = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/blender_3d/blender_test_image'
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


def get_3x4_RT_matrix_from_blender(obj):
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


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)[0]
    return K @ RT


image_file_path = base_file_path + '.png'
blend_file_path = base_file_path + '.blend'

exclude_object_names = ["Camera", "Cube", "Light"]
delete_all_objects_except(exclude_object_names)

scene = bpy.context.scene
cam = bpy.data.objects['Camera']
obj = bpy.data.objects['Cube']

scene.render.resolution_x = 1280
scene.render.resolution_y = 960
width = scene.render.resolution_x
height = scene.render.resolution_y

limit = 0.1
DeselectEdgesAndPolygons(obj)

mWorld = obj.matrix_world
vertices = [mWorld @ v.co for v in obj.data.vertices]

# Create a new material with an emission shader
material = bpy.data.materials.new(name="Emission")
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
material.node_tree.nodes.remove(bsdf)
emission = material.node_tree.nodes.new('ShaderNodeEmission')
output = material.node_tree.nodes["Material Output"]
material.node_tree.links.new(output.inputs[0], emission.outputs[0])
emission.inputs[1].default_value = 10  # Increase the strength of the emission

print('-------------------')
points3D = []
for i, v in enumerate(vertices):
    co2D = world_to_camera_view(scene, cam, v)

    # Add a small sphere at each vertex position
    bpy.ops.mesh.primitive_uv_sphere_add(location=v)
    bpy.ops.transform.resize(value=(0.005, 0.005, 0.005))
    sphereObject = bpy.context.object

    # Assign the emission material to the sphere
    if sphereObject.data.materials:
        sphereObject.data.materials[0] = material
    else:
        sphereObject.data.materials.append(material)

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



# use generator expressions () or list comprehensions []
verts = (vert.co for vert in obj.data.vertices)
coords_2d = [world_to_camera_view(scene, cam, coord) for coord in verts]

# 2d data printout:
rnd = lambda i: round(i)
for x, y, distance_to_lens in coords_2d:
    print("{},{}".format(float(width*x), float(height*y)))

scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = image_file_path
bpy.ops.render.render(write_still=1)

bpy.ops.wm.save_as_mainfile(filepath=blend_file_path)

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
P = get_3x4_P_matrix_from_blender(cam)
projectionMatrix = np.matrix(P)
print('projection M', projectionMatrix)

# print R|T
_, rvec, tvec = get_3x4_RT_matrix_from_blender(cam)
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