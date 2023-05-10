bl_info = {
    "name": "Circle LEDs Addon",
    "blender": (2, 80, 0),
    "category": "Object",
}

import bpy
import pickle
import gzip
from mathutils import Vector, geometry
import platform

text_objects_hidden = True

def create_text_objects(led_data, led_size, shape):
    text_distance = led_size
    for i, coord in enumerate(led_data):
        normalized_direction = Vector(coord).normalized()

        bpy.ops.object.text_add()
        text_obj = bpy.context.active_object
        text_obj.name = f"LED_Number_{i}"
        text_obj.data.body = str(i)

        if 'cylinder' in shape:
            text_location = [coord[0] + text_distance * normalized_direction.x,
                            coord[1] + text_distance * normalized_direction.y,
                            coord[2]]
        else:
            text_location = [coord[0] + text_distance * normalized_direction.x,
                            coord[1] + text_distance * normalized_direction.y,
                            coord[2] + text_distance * normalized_direction.z]
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

    shape: bpy.props.StringProperty(
        name="Shape",
        description="Enter the shape",
        default="cylinder_base"
    )

    def execute(self, context):
        global text_objects_hidden

        pickle_file = None
        os_name = platform.system()
        print('shape', self.shape)
        if os_name == 'Windows':
            print("This is Windows")
            if self.shape == 'sphere':
                pickle_file = 'D:/OpenCV_APP/led_pos_simulation/find_pos_legacy/result.pickle'
            elif self.shape == 'cylinder':
                pickle_file = 'D:/OpenCV_APP/led_pos_simulation/find_pos_legacy/result_cylinder.pickle'
        elif os_name == 'Linux':
            print("This is Linux")            
            if self.shape == 'sphere':
                pickle_file = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/find_pos_legacy/result.pickle'
            elif self.shape == 'cylinder':
                pickle_file = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/find_pos_legacy/result_cylinder.pickle'
            elif self.shape == 'cylinder_base':
                pickle_file = '/home/rangkast.jeong/Project/OpenCV_APP/led_pos_simulation/find_pos_legacy/result_cylinder_base.pickle'
        else:
            print("Unknown OS")



        with gzip.open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        led_data = data['LED_INFO']
        led_size = 0.003

        if not any(obj.type == 'FONT' and obj.name.startswith("LED_Number_") for obj in bpy.data.objects):
            create_text_objects(led_data, led_size, self.shape)

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

        # 입력창 추가
        shape_row = layout.row()
        shape_row.prop(context.scene, "circle_leds_shape", text="Shape")

        # 버튼 추가
        create_circle_leds_op = layout.operator("object.create_circle_leds", text="Show LEDs Number")
        create_circle_leds_op.shape = context.scene.circle_leds_shape

def register():
    bpy.utils.register_class(CircleLEDsOperator)
    bpy.utils.register_class(CircleLEDsPanel)

    bpy.types.Scene.circle_leds_shape = bpy.props.StringProperty(
        name="Shape",
        description="Enter the shape",
        default="cylinder_base"
    )

def unregister():
    bpy.utils.unregister_class(CircleLEDsOperator)
    bpy.utils.unregister_class(CircleLEDsPanel)

    del bpy.types.Scene.circle_leds_shape

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent))
    register()

