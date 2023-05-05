bl_info = {
    "name": "Rotate Object",
    "blender": (2, 80, 0),
    "category": "Object",
}

import bpy
from math import radians

class OBJECT_OT_rotate(bpy.types.Operator):
    bl_idname = "object.rotate_operator"
    bl_label = "Rotate Object"
    bl_options = {"REGISTER", "UNDO"}

    axis: bpy.props.EnumProperty(
        items=[("X", "X", ""), ("Y", "Y", ""), ("Z", "Z", "")],
        name="Axis",
        default="Z",
    )

    angle: bpy.props.FloatProperty(
        name="Angle (degrees)",
        default=45.0,
        min=0.0,
        max=360.0,
    )

    def execute(self, context):
        obj = context.active_object

        if obj is None:
            self.report({"ERROR"}, "No active object")
            return {"CANCELLED"}

        rotation_angle = radians(self.angle)
        bpy.ops.transform.rotate(value=rotation_angle, orient_axis=self.axis)

        return {"FINISHED"}


class OBJECT_PT_rotate_panel(bpy.types.Panel):
    bl_label = "Rotate Object"
    bl_idname = "OBJECT_PT_rotate_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Tools"

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.prop(context.scene, "rotation_axis")

        row = layout.row()
        row.prop(context.scene, "rotation_angle")

        row = layout.row()
        row.operator("object.rotate_operator")


def register():
    bpy.utils.register_class(OBJECT_OT_rotate)
    bpy.utils.register_class(OBJECT_PT_rotate_panel)

    bpy.types.Scene.rotation_axis = bpy.props.EnumProperty(
        items=[("X", "X", ""), ("Y", "Y", ""), ("Z", "Z", "")],
        name="Axis",
        default="Z",
    )

    bpy.types.Scene.rotation_angle = bpy.props.FloatProperty(
        name="Angle (degrees)",
        default=45.0,
        min=0.0,
        max=360.0,
    )


def unregister():
    bpy.utils.unregister_class(OBJECT_OT_rotate)
    bpy.utils.unregister_class(OBJECT_PT_rotate_panel)

    del bpy.types.Scene.rotation_axis
    del bpy.types.Scene.rotation_angle


if __name__ == "__main__":
    register()
