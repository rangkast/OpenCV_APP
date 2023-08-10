from Advanced_Function import *

'''
ORIGINAL 3D Points
'''
# MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/arcturas_right_1_self.json"))
MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/arcturas_left_1_self.json"))
# MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/rifts_left_2.json"))
# MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/rifts_right_9.json"))
# MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/semi_slam_plane.json"))
# MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/semi_slam_curve.json"))
# MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/semi_slam_polyhedron.json"))

poly_hedron_cadidates = [
[-0.06308038, 0.01984174, -0.02024762],
[-0.04824327, 0.00533525, -0.01977697],
[-0.01890498, 0.01978277, -0.0051605],
[-0.00201741, 0.00495996, -0.00489448],
[0.01838035, 0.02010325, -0.0046952],
[0.04961416, 0.01984763, -0.0200507],
[0.06292902, 0.00512941, -0.02017453],
]

curve_candidates = [
[-0.06459047, 0.02002626, -0.03601638],
[-0.05198585, 0.0050076, -0.02130529],
[-0.03051528, 0.01999273, -0.00731907],
[-0.0104295, 0.00495597, -0.00155286],
[0.01470543, 0.0200651, -0.00227665],
[0.03910384, 0.00499392, -0.01158884],
[0.05206699, 0.01994052, -0.02125366],
[0.06446175, 0.0050179, -0.03607852],
]

plane_candidates = [
[-0.0630138, 0.01960507, -0.01325],
[-0.04843088, 0.00538483, -0.01325],
[-0.01995668, 0.01994691, -0.01325],
[-0.0019146, 0.00521891, -0.01325],
[0.01933924, 0.01994614, -0.01325],
[0.04776097, 0.00499845, -0.01325],
[0.06305025, 0.01989964, -0.01325],
]

BLOB_CNT = len(MODEL_DATA)
print('PTS')
for i, leds in enumerate(MODEL_DATA):
    print(f"{np.array2string(leds, separator=', ')},")
print('DIR')
for i, dir in enumerate(DIRECTION):
    print(f"{np.array2string(dir, separator=', ')},")
    
show_calibrate_data(np.array(MODEL_DATA), np.array(DIRECTION), TARGET=np.array(plane_candidates))      