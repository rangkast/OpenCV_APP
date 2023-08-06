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
import random
from mathutils import Quaternion
import sys
print(sys.exec_prefix)

# "functions.py" 텍스트를 모듈로 가져옵니다.
functions = bpy.data.texts["functions.py"].as_module()
# definitions 모듈 내의 모든 이름을 가져옵니다.
names = dir(functions)
# 모든 이름에 대해 전역 변수를 설정합니다.
globals().update({name: getattr(functions, name) for name in names})

os_name = platform.system()
if os_name == 'Windows':
    print("This is Windows")
    BA_3D_PATH = "D:/OpenCV_APP/BA_3D.pickle"
    REMADE_3D_INFO_PATH = "D:/OpenCV_APP/REMADE_3D_INFO.pickle"
    RIGID_3D_TRANSFORM_PATH = "D:/OpenCV_APP/RIGID_3D_TRANSFORM.pickle"
    CAMERA_INFO_PATH = "D:/OpenCV_APP/CAMERA_INFO.pickle"
#    CAMERA_INFO_PATH = "D:/OpenCV_APP/FINAL_CAMERA_INFO.pickle"
    FITTING_CIRCLE_PATH = "D:/OpenCV_APP/FITTING_CIRCLE.pickle"
elif os_name == 'Linux':
    print("This is Linux")
    BA_3D_PATH = "/home/rangkast.jeong/Project/OpenCV_APP/BA_3D.pickle"
    REMADE_3D_INFO_PATH = "/home/rangkast.jeong/Project/OpenCV_APP/REMADE_3D_INFO.pickle"
    RIGID_3D_TRANSFORM_PATH = "/home/rangkast.jeong/Project/OpenCV_APP/RIGID_3D_TRANSFORM.pickle"
    CAMERA_INFO_PATH = "/home/rangkast.jeong/Project/OpenCV_APP/CAMERA_INFO.pickle"
    FITTING_CIRCLE_PATH = "/home/rangkast.jeong/Project/OpenCV_APP/FITTING_CIRCLE.pickle"
else:
    print("Unknown OS")
    
    
delete_all_objects_except(exclude_object_names)

FITTING_CIRCLE = pickle_data(READ, FITTING_CIRCLE_PATH, None)['P_fitcircle']
#CAMERA_INFO = pickle_data(READ, CAMERA_INFO_PATH, None)['FINAL_CAMERA_INFO']
CAMERA_INFO = pickle_data(READ, CAMERA_INFO_PATH, None)['CAMERA_INFO']
BA_3D = pickle_data(READ, BA_3D_PATH, None)['BA_3D']
BA_3D_LED_INDICIES = pickle_data(READ, BA_3D_PATH, None)['LED_INDICES']
REMADE_3D_INFO_O = pickle_data(READ, REMADE_3D_INFO_PATH, None)['REMADE_3D_INFO_O']
RIGID_3D_TRANSFORM__PCA = pickle_data(READ, RIGID_3D_TRANSFORM_PATH, None)['PCA_ARRAY_LSM']
RIGID_3D_TRANSFORM__IQR = pickle_data(READ, RIGID_3D_TRANSFORM_PATH, None)['IQR_ARRAY_LSM']

def draw_objects(data, shape, radius, color, use_emission=False, use_simplify=False):
    # Set the render engine to 'BLENDER_EEVEE'
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    # Enable Simplify and lower render quality settings if use_simplify is True
    if use_simplify:
        bpy.context.scene.render.use_simplify = True
        bpy.context.scene.render.simplify_subdivision = 0
        bpy.context.scene.render.simplify_child_particles = 0
        bpy.context.scene.render.simplify_gpencil = 1
        
        # Lower resolution
        bpy.context.scene.render.resolution_x = 640  # For example, set to 640x480
        bpy.context.scene.render.resolution_y = 480

        # Lower sampling
        bpy.context.scene.eevee.taa_render_samples = 32  # Lower number of samples
    else:
        bpy.context.scene.render.use_simplify = False

    # Create a new material
    material = bpy.data.materials.new(name="ColoredMaterial")
    material.use_nodes = True

    # Get material node tree
    nodes = material.node_tree.nodes

    # Clear all nodes
    for node in nodes:
        nodes.remove(node)

    if use_emission:
        # Add emission node
        emission = nodes.new(type='ShaderNodeEmission')
        emission.inputs["Strength"].default_value = 1.0
        emission.inputs["Color"].default_value = (*color, 1)  # input color

        # Add output node
        output = nodes.new(type='ShaderNodeOutputMaterial')

        # Link nodes
        links = material.node_tree.links
        link = links.new(emission.outputs["Emission"], output.inputs["Surface"])

    else:
        # Add Principled BSDF node
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.inputs["Base Color"].default_value = (*color, 1)  # input color

        # Add output node
        output = nodes.new(type='ShaderNodeOutputMaterial')

        # Link nodes
        links = material.node_tree.links
        link = links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Create object at each point with specified radius/size and assign the material
    for point in data:
        if shape == 'sphere':
            bpy.ops.mesh.primitive_uv_sphere_add(location=point, radius=radius)
        elif shape == 'cube':
            bpy.ops.mesh.primitive_cube_add(location=point, size=2*radius)  # Cube size is edge length, so use 2*radius

        obj = bpy.context.object
        obj.data.materials.append(material)

def draw_arrow(position, direction, length, material, radius=0.0005):
    """
    Draw an arrow to represent a vector.
    
    Parameters:
    position: The starting point of the arrow
    direction: The direction of the arrow
    length: The length of the arrow
    material: The material of the arrow
    radius: The radius of the arrow (default is 0.0005)
    """
    # Normalize the direction
    direction_norm = direction.normalized()
    # Calculate the end point of the arrow
    end_point = position + length * direction_norm
    # Draw a cylinder from the starting point to the end point
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=length,
        location=(position + end_point) / 2,  # The center of the cylinder
    )
    # Get the cylinder object
    cylinder = bpy.context.object
    # Rotate the cylinder to align with the direction
    rot_quat = direction_norm.to_track_quat('Z', 'Y')
    cylinder.rotation_euler = rot_quat.to_euler()
    # Assign the material to the cylinder
    cylinder.data.materials.append(material)

def draw_camera_pos_and_dir():
     # Create a new material
    omat = bpy.data.materials.new(name="RedMaterial")
    omat.diffuse_color = (1.0, 0.0, 0.0, 1.0) 
    bmat = bpy.data.materials.new(name="BlueMaterial")
    bmat.diffuse_color = (0.0, 0.0, 1.0, 1.0)
    bamat = bpy.data.materials.new(name="YellowMaterial")
    bamat.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    rgid_mat = bpy.data.materials.new(name="GreenMaterial")
    rgid_mat.diffuse_color = (0.0, 1.0, 0.0, 1.0)
    opositions = []
    bpositions = []
    bapositions = []
    for key, camera_info in CAMERA_INFO.items():
#        print('frame_cnt: ', key)
        orvec = camera_info['OPENCV']['rt']['rvec']
        otvec = camera_info['OPENCV']['rt']['tvec']
        degree = camera_info['ANGLE']
#        brvec = camera_info['BLENDER']['rt']['rvec']
#        btvec = camera_info['BLENDER']['rt']['tvec']
#        barvec = camera_info['BA_RT']['rt']['rvec']
#        batvec = camera_info['BA_RT']['rt']['tvec']
        
        
          
        if len(orvec) > 0 and degree == 0:
            # Add Camera Position
            oposition, orotation_quat = blender_location_rotation_from_opencv(orvec, otvec)        
            # Convert quaternion rotation to Euler rotation
            rotation_mat = orotation_quat.to_matrix().to_4x4()
            rotation_euler = rotation_mat.to_euler()        
            # Create a small sphere at the camera position
            bpy.ops.mesh.primitive_uv_sphere_add(location=oposition, radius=0.001)
            osphere = bpy.context.object
            osphere.data.materials.append(omat)  # Assign the material to the sphere
            opositions.append(np.array(oposition))
            # For OpenCV Camera
            draw_arrow(oposition, orotation_quat @ mathutils.Vector((0.0, 0.0, -1.0)), 0.03, omat)
            bpy.ops.object.text_add(location=(oposition[0], oposition[1], oposition[2] + 0.005))  # adjust z-coordinate to place the text above the sphere
            txt_obj = bpy.context.object
            txt_obj.data.body = str(key)
            txt_obj.rotation_euler = rotation_euler  # Optional, if you want to align text with the sphere's orientation
            txt_obj.scale = (0.005, 0.005, 0.005)  # adjust the values to get the desired size

    data = OrderedDict()
    data['opositions'] = opositions
    pickle_data(WRITE, 'D:/OpenCV_APP/BLENDER.pickle', data)
#    pickle_data(WRITE, '/home/rangkast.jeong/Project/OpenCV_APP/BLENDER.pickle', data)
            
            
make_models('real')

#OPENCV_REMAKE_3D = []
#for blob_id, data_list in REMADE_3D_INFO_O.items():
#    for data in data_list:
#        point_3d = data.reshape(-1)
#        OPENCV_REMAKE_3D.append(point_3d)

#draw_objects(origin_led_data, 'sphere', 0.0003, (1, 1, 1), use_emission=True, use_simplify=False)
#draw_objects(target_led_data, 'sphere', 0.0003, (0, 0, 0), use_emission=False, use_simplify=False)

#draw_objects(BA_3D, 'cube', 0.0001, (0, 0, 1), use_emission=True, use_simplify=True)
#draw_objects(OPENCV_REMAKE_3D, 'cube', 0.0001, (1, 0, 0), use_emission=True, use_simplify=True)

#draw_objects(RIGID_3D_TRANSFORM__PCA, 'sphere', 0.0003, (1, 1, 0), use_emission=False, use_simplify=False)
#draw_objects(RIGID_3D_TRANSFORM__IQR, 'sphere', 0.0003, (0, 1, 1), use_emission=False, use_simplify=False)

draw_camera_pos_and_dir()

print('#### DONE ####')