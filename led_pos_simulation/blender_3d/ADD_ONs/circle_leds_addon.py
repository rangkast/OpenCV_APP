bl_info = {
    "name": "Circle LEDs Addon",
    "blender": (2, 80, 0),
    "category": "Object",
}

import bpy
import pickle
import gzip
from mathutils import Vector, geometry

text_objects_hidden = True

def create_text_objects(led_data, led_size):
    text_distance = led_size * -3
    for i, coord in enumerate(led_data):
        normalized_direction = Vector(coord).normalized()

        bpy.ops.object.text_add()
        text_obj = bpy.context.active_object
        text_obj.name = f"LED_Number_{i}"
        text_obj.data.body = str(i)

        text_location = [coord[0] - text_distance * normalized_direction.x,
                        coord[1] - text_distance * normalized_direction.y,
                        coord[2] - text_distance * normalized_direction.z]
        text_obj.location = text_location
        text_obj.rotation_euler = (0, 0, 0)
        text_obj.rotation_mode = 'QUATERNION'
        text_obj.rotation_quaternion = normalized_direction.to_track_quat('Z', 'Y')
        text_obj.scale = (0.01, 0.01, 0.01)

        text_material = bpy.data.materials.new(name=f"Text_Material_{i}")
        text_material.use_nodes = False
        text_material.diffuse_color = (0, 0, 0, 1)

        text_obj.data.materials.append(text_material)

        text_obj.hide_viewport = False
        text_obj.hide_render = True

class CircleLEDsOperator(bpy.types.Operator):
    bl_idname = "object.create_circle_leds"
    bl_label = "Create Circle LEDs"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        global text_objects_hidden

        pickle_file = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/find_pos_legacy/result.pickle'

        with gzip.open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        led_data = data['LED_INFO']
        led_size = 0.002

        if not any(obj.type == 'FONT' and obj.name.startswith("LED_Number_") for obj in bpy.data.objects):
            create_text_objects(led_data, led_size)

        text_objects_hidden = not text_objects_hidden

        for obj in bpy.data.objects:
            if obj.type == 'FONT' and obj.name.startswith("LED_Number_"):
                obj.hide_viewport = text_objects_hidden
                obj.hide_render = text_objects_hidden

        return {'FINISHED'}


class CircleLEDsPanel(bpy.types.Panel):
    bl_label = "Circle LEDs"
    bl_idname = "OBJECT_PT_circle_leds"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tools"

    def draw(self, context):
        layout = self.layout
        layout.operator("object.create_circle_leds", text="Show LEDs Number")

def register():
    bpy.utils.register_class(CircleLEDsOperator)
    bpy.utils.register_class(CircleLEDsPanel)

def unregister():
    bpy.utils.unregister_class(CircleLEDsOperator)
    bpy.utils.unregister_class(CircleLEDsPanel)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent))
    register()

