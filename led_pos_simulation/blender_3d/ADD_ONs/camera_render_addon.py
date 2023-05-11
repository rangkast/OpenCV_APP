bl_info = {
    "name": "Camera Render Addon",
    "blender": (2, 80, 0),
    "category": "Object",
}


import bpy
import os
from datetime import datetime
from math import degrees
from mathutils import Matrix, Vector
import numpy as np
import cv2

camera_matrix = [
    # cam 0
    [np.array([[712.623, 0.0, 653.448],
               [0.0, 712.623, 475.572],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.072867], [-0.026268], [0.007135], [-0.000997]], dtype=np.float64)],

    # cam 1
    [np.array([[716.896, 0.0, 668.902],
               [0.0, 716.896, 460.618],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.07542], [-0.026874], [0.006662], [-0.000775]], dtype=np.float64)]
]
default_dist_coeffs = np.zeros((4, 1))



# BKE_camera_sensor_size
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
    
    # Calculate the image center in pixels
    cscale = resolution_x_in_px / sensor_size_in_mm
    cx = camd.shift_x * s_u + resolution_x_in_px / 2.0
    cy = camd.shift_y * s_v + resolution_y_in_px / 2.0
    cx_px  = (cx - resolution_x_in_px / 2.0) * cscale
    cy_px  = -(cy - resolution_y_in_px / 2.0) * cscale
    
    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    print('sensor_fit', sensor_fit)
    print('f_in_mm', f_in_mm)
    print('sensor_size_in_mm', sensor_size_in_mm)
    print('res_x res_y', resolution_x_in_px, resolution_y_in_px)
    print('pixel_aspect_ratio', pixel_aspect_ratio)
    print('shift_x shift_y', camd.shift_x, camd.shift_y)
    print('K', K)
    return K

def blender_location_rotation_from_opencv(rvec, tvec, obj):
    isCamera = (obj.type == 'CAMERA')
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

def get_3x4_RT_matrix_from_blender(obj):
    isCamera = (obj.type == 'CAMERA')
    R_BlenderView_to_OpenCVView = np.diag([1 if isCamera else -1,-1,-1])
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

def export_camera_to_opencv(cam_name, path, file_name):
    cam = bpy.data.objects[cam_name]
    P = get_3x4_P_matrix_from_blender(cam)

    nP = np.matrix(P)
    # path = bpy.path.abspath("//")
    filename = file_name + ".txt"
    file = path + "/"+ filename
    np.savetxt(file, nP)
    print(f"Saved to: \"{file}\".")

def export_object_location_to_opencv(obj_name):
    obj = bpy.data.objects[obj_name]
    nT = get_3x4_RT_matrix_from_blender(obj)[2]

    path = bpy.path.abspath("//")
    file = f"{path}{obj.name}.txt"
    np.savetxt(file, nT)
    print(f"Saved to: \"{file}\".")


def set_mesh_objects_opaque():
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            for slot in obj.material_slots:
                if slot.material:
                    if slot.material.use_nodes:
                        nodes = slot.material.node_tree.nodes
                        principled_bsdf = None
                        for node in nodes:
                            if node.type == 'BSDF_PRINCIPLED':
                                principled_bsdf = node
                                break
                        if principled_bsdf:
                            principled_bsdf.inputs['Alpha'].default_value = 1



def save_viewport_render(context, filepath):
    # 뷰포트 렌더링 설정
    render = context.scene.render
    render.image_settings.file_format = 'PNG'
    render.use_file_extension = True
    render.filepath = filepath

    # 뷰포트 렌더링 및 저장
    bpy.ops.render.opengl(write_still=True)

def get_objects_with_name(name_pattern):
    
    for obj in bpy.data.objects:
        if name_pattern in obj.name:
            return obj.name
            
    return None


def get_suffix_from_name(name, delimiter='_'):
    if delimiter not in name:
        raise ValueError(f"The delimiter '{delimiter}' is not present in the name '{name}'")
    
    return name.split(delimiter)[-1]


class ButtonAOperator(bpy.types.Operator):
    bl_idname = "object.button_a"
    bl_label = "Button A"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        camera_name = context.scene.camera_input  # 사용자 입력으로 카메라 이름 가져오기
        if camera_name not in bpy.data.objects or bpy.data.objects[camera_name].type != 'CAMERA':
            self.report({'ERROR'}, f"No camera found with name {camera_name}")
            return {'CANCELLED'}

        # camera_names = ["CAMERA_0", "CAMERA_1"]
        print('camera render view')

        name_pattern = "MeshObject"
        matched_objects = get_objects_with_name(name_pattern)
        suffix = get_suffix_from_name(matched_objects)
        print('suffix', suffix)

        # for cam_id, camera_name in enumerate(camera_names):
        # 카메라 객체를 선택
        try:
            camera = bpy.data.objects[camera_name]
            print('camera', camera)
            # 카메라의 위치와 회전 값을 가져오기
            location = camera.location
            rotation = camera.rotation_euler
            quat = camera.rotation_quaternion
            # XYZ 오일러 각도를 degree 단위로 변환
            rotation_degrees = tuple(degrees(angle) for angle in rotation)
            # 결과 출력
            print(f"{camera_name} 위치: ", location)
            print(f"{camera_name} XYZ 오일러 회전 (도): ", rotation_degrees)
            print(f"{camera_name} XYZ 쿼너티온: ", quat)

            # 렌더링 설정
            scene = bpy.context.scene
            scene.render.engine = 'BLENDER_EEVEE'  # 'CYCLES' 또는 'BLENDER_EEVEE'로 변경 가능
            scene.camera = camera  # 현재 씬의 카메라로 설정

            print(f"Current camera for rendering is {scene.camera.name}")  # 현재 렌더링 카메라 출력


            # 렌더링 결과를 저장할 경로 지정
            output_path = os.path.join(bpy.path.abspath("//"), 'render_output')
            os.makedirs(output_path, exist_ok=True)

            # 현재 시간에 대한 타임스탬프 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 렌더링 결과 이미지 파일명 설정
            filename = f"{camera_name}_{suffix}_x{round(rotation_degrees[0], 1)}_y{round(rotation_degrees[1], 1)}_z{round(rotation_degrees[2], 1)}_{timestamp}"
            scene.render.filepath = os.path.join(output_path, filename + ".png")

            # path = bpy.path.abspath("//")
            # file = path + filename
            
            # 렌더링 수행
            bpy.ops.render.render(write_still=True)
            
            export_camera_to_opencv(camera_name, output_path, filename)
            # export_object_location_to_opencv('MeshObject')
        except KeyError:
            print('camera not found', camera_name)            

        return {'FINISHED'}


class ButtonBOperator(bpy.types.Operator):
    bl_idname = "object.button_b"
    bl_label = "Button B"
    bl_options = {'REGISTER', 'UNDO'}

    
    def execute(self, context):
            camera_name = context.scene.camera_input  # 사용자 입력으로 카메라 이름 가져오기
            if camera_name not in bpy.data.objects or bpy.data.objects[camera_name].type != 'CAMERA':
                self.report({'ERROR'}, f"No camera found with name {camera_name}")
                return {'CANCELLED'}
            # camera_names = ["CAMERA_0", "CAMERA_1"]
            print('material preview') 
            name_pattern = "MeshObject"
            matched_objects = get_objects_with_name(name_pattern)
            suffix = get_suffix_from_name(matched_objects)
            print('suffix', suffix)

            # for cam_id, camera_name in enumerate(camera_names):
            # 카메라 객체를 선택
            try:
                camera = bpy.data.objects[camera_name]
                print('camera', camera)
                # 카메라의 위치와 회전 값을 가져오기
                location = camera.location
                rotation = camera.rotation_euler
                quat = camera.rotation_quaternion
                # XYZ 오일러 각도를 degree 단위로 변환
                rotation_degrees = tuple(degrees(angle) for angle in rotation)
                # 결과 출력
                print(f"{camera_name} 위치: ", location)
                print(f"{camera_name} XYZ 오일러 회전 (도): ", rotation_degrees)
                print(f"{camera_name} XYZ 쿼너티온: ", quat)

                # 현재 씬의 카메라로 설정
                context.scene.camera = camera  

                # 뷰포트 쉐이딩 설정
                shading = context.space_data.shading
                shading.type = 'MATERIAL'  # 머티리얼 프리뷰 모드로 설정

                # 렌더링 결과를 저장할 경로 지정
                output_path = os.path.join(bpy.path.abspath("//"), 'render_output')
                os.makedirs(output_path, exist_ok=True)

                # 현재 시간에 대한 타임스탬프 생성
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 렌더링 결과 이미지 파일명 설정
                filename = f"{camera_name}_{suffix}_MATERIAL_x{int(rotation_degrees[0])}_y{int(rotation_degrees[1])}_z{int(rotation_degrees[2])}_{timestamp}"
                filepath = os.path.join(output_path, filename + ".png")

                # 뷰포트 렌더링 및 저장
                save_viewport_render(context, filepath)
                # export_camera_to_opencv(camera_name, output_path, filename)
                # export_object_location_to_opencv('MeshObject')
            except KeyError:
                print('camera not found', camera_name)            

            return {'FINISHED'}



class OpenCVPanel(bpy.types.Panel):
    bl_label = "Camera Render"
    bl_idname = "OBJECT_PT_opencv_example"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tools"

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "camera_input")  # 사용자 입력 필드를 패널에 추가
        layout.operator("object.button_a", text="render view")
        layout.operator("object.button_b", text="material view")

def register():
    bpy.utils.register_class(ButtonAOperator)
    bpy.utils.register_class(ButtonBOperator)
    bpy.utils.register_class(OpenCVPanel)
    bpy.types.Scene.camera_input = bpy.props.StringProperty(name="Camera Name")  # 새로운 속성 추가

def unregister():
    bpy.utils.unregister_class(ButtonAOperator)
    bpy.utils.unregister_class(ButtonBOperator)
    bpy.utils.unregister_class(OpenCVPanel)
    del bpy.types.Scene.camera_input  # 속성 제거

if __name__ == "__main__":
    register()