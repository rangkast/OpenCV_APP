bl_info = {
    "name": "OpenCV Example Addon",
    "blender": (2, 80, 0),
    "category": "Object",
}

import bpy
import pickle
import gzip
from mathutils import Vector, geometry
import cv2


class ButtonAOperator(bpy.types.Operator):
    bl_idname = "object.button_a"
    bl_label = "Button A"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        print("Button A pressed")
        return {'FINISHED'}

class ButtonBOperator(bpy.types.Operator):
    bl_idname = "object.button_b"
    bl_label = "Button B"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        print("Button B pressed")
        return {'FINISHED'}

class OpenCVPanel(bpy.types.Panel):
    bl_label = "OpenCV Example"
    bl_idname = "OBJECT_PT_opencv_example"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "OpenCV"

    def draw(self, context):
        layout = self.layout
        layout.operator("object.button_a", text="A")
        layout.operator("object.button_b", text="B")

def register():
    bpy.utils.register_class(ButtonAOperator)
    bpy.utils.register_class(ButtonBOperator)
    bpy.utils.register_class(OpenCVPanel)

def unregister():
    bpy.utils.unregister_class(ButtonAOperator)
    bpy.utils.unregister_class(ButtonBOperator)
    bpy.utils.unregister_class(OpenCVPanel)

if __name__ == "__main__":
    register()
