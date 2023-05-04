bl_info = {
    "name": "Camera Rendering Addon",
    "blender": (2, 80, 0),
    "category": "Object",
}

import bpy
import os
from datetime import datetime
from math import degrees

class CameraRenderingOperator(bpy.types.Operator):
    bl_idname = "object.camera_rendering"
    bl_label = "Render from Cameras"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        camera_names = ["CAMERA_0", "CAMERA_1"]

        for camera_name in camera_names:
            # 카메라 객체를 선택
            camera = bpy.data.objects[camera_name]
            # 카메라의 위치와 회전 값을 가져오기
            location = camera.location
            rotation = camera.rotation_euler
            # XYZ 오일러 각도를 degree 단위로 변환
            rotation_degrees = tuple(degrees(angle) for angle in rotation)
            # 결과 출력
            print(f"{camera_name} 위치: ", location)
            print(f"{camera_name} XYZ 오일러 회전 (도): ", rotation_degrees)

            # 렌더링 설정
            scene = bpy.context.scene
            scene.camera = camera  # 현재 씬의 카메라로 설정
            # 렌더링 결과를 저장할 경로 지정
            output_path = os.path.join(os.path.expanduser('~'), 'render_output')
            os.makedirs(output_path, exist_ok=True)

            # 현재 시간에 대한 타임스탬프 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 렌더링 결과 이미지 파일명 설정
            filename = f"{camera_name}_x{int(rotation_degrees[0])}_y{int(rotation_degrees[1])}_z{int(rotation_degrees[2])}_{timestamp}.png"
            scene.render.filepath = os.path.join(output_path, filename)

            # 렌더링 수행
            bpy.ops.render.render(write_still=True)

        return {'FINISHED'}

class CameraRenderingPanel(bpy.types.Panel):
    bl_label = "Camera Rendering"
    bl_idname = "OBJECT_PT_camera_rendering"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tools"

    def draw(self, context):
        layout = self.layout
        layout.operator("object.camera_rendering", text="Render from Cameras")

def register():
    bpy.utils.register_class(CameraRenderingOperator)
    bpy.utils.register_class(CameraRenderingPanel)

def unregister():
    bpy.utils.unregister_class(CameraRenderingOperator)
    bpy.utils.unregister_class(CameraRenderingPanel)

if __name__ == "__main__":
    register()
