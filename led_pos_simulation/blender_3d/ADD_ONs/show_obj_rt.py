import bpy
import numpy as np
from mathutils import Matrix, Vector
import cv2

bl_info = {
    "name": "RT Matrix Info",
    "author": "Your Name",
    "version": (1, 0),
    "blender": (2, 83, 0),
    "location": "View3D > Sidebar > Your Tab",
    "description": "Displays RT matrix info",
    "warning": "",
    "doc_url": "",
    "category": "3D View",
}

class RT_OT_MatrixInfo(bpy.types.Operator):
    bl_idname = "rt.matrix_info"
    bl_label = "Get RT Matrix Info"
    
    def execute(self, context):
        obj = context.object
        get_3x4_RT_matrix_from_blender(obj)
        return {'FINISHED'}

class RT_OT_BlenderLocRotFromOpenCV(bpy.types.Operator):
    bl_idname = "rt.blender_locrot_from_opencv"
    bl_label = "Get Blender LocRot from OpenCV"

    rvec: bpy.props.StringProperty(name="RVec", default="")
    tvec: bpy.props.StringProperty(name="TVec", default="")
    isCamera: bpy.props.BoolProperty(name="Is Camera", default=True)
    
    def execute(self, context):
        rvec = np.fromstring(self.rvec, sep=',')
        tvec = np.fromstring(self.tvec, sep=',')
        location, rotation = blender_location_rotation_from_opencv(rvec, tvec, self.isCamera)
        print("Location:", location)
        print("Rotation:", rotation)
        return {'FINISHED'}
    
class RT_PT_MatrixInfo(bpy.types.Panel):
    bl_label = "RT Matrix Info"
    bl_idname = "RT_PT_matrix_info"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Your Tab'
    
    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.operator("rt.matrix_info")

        box = layout.box()
        box.label(text="Blender LocRot from OpenCV")
        row = box.row()
        op = row.operator("rt.blender_locrot_from_opencv")
        row = box.row()
        row.prop(op, "rvec")
        row = box.row()
        row.prop(op, "tvec")
        row = box.row()
        row.prop(op, "isCamera")

def get_3x4_RT_matrix_from_blender(obj):
    isCamera = (obj.type == 'CAMERA')
    R_BlenderView_to_OpenCVView = np.diag([1 if isCamera else -1,-1,-1])
    location, rotation = obj.matrix_world.decompose()[:2]
    axis, angle = rotation.to_axis_angle()
    R_BlenderView = Matrix.Rotation(angle, 3, axis).transposed()
    T_BlenderView = -1.0 * R_BlenderView @ location
    R_OpenCV = R_BlenderView_to_OpenCVView @ R_BlenderView
    T_OpenCV = R_BlenderView_to_OpenCVView @ T_BlenderView
    R, _ = cv2.Rodrigues(R_OpenCV)
    print('R_OpenCV', R_OpenCV)
    print('R_OpenCV(Rod)', R.ravel())
    print('T_OpenCV', T_OpenCV)

def blender_location_rotation_from_opencv(rvec, tvec, isCamera=True):
    R_BlenderView_to_OpenCVView = Matrix([
        [1 if isCamera else -1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ])
    R_OpenCV, _ = cv2.Rodrigues(rvec)
    R_BlenderView = R_BlenderView_to_OpenCVView @ Matrix(R_OpenCV.tolist())
    T_BlenderView = R_BlenderView_to_OpenCVView @ Vector(tvec)
    R_BlenderView_inv = R_BlenderView.transposed()
    location = -1.0 * R_BlenderView_inv @ T_BlenderView
    rotation = R_BlenderView_inv.to_quaternion()
    return location, rotation

def register():
    bpy.utils.register_class(RT_OT_MatrixInfo)
    bpy.utils.register_class(RT_OT_BlenderLocRotFromOpenCV)
    bpy.utils.register_class(RT_PT_MatrixInfo)

def unregister():
    bpy.utils.unregister_class(RT_OT_MatrixInfo)
    bpy.utils.unregister_class(RT_OT_BlenderLocRotFromOpenCV)
    bpy.utils.unregister_class(RT_PT_MatrixInfo)

if __name__ == "__main__":
    register()
