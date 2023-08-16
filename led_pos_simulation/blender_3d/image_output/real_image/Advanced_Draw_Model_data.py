from Advanced_Function import *

'''
ORIGINAL 3D Points
'''

# MODEL_PATH = f"{script_dir}/jsons/specs/arcturus_#3_right+.json"
# MODEL_PATH_SAVE = f"{script_dir}/jsons/specs/arcturas_#3_right_new.json"

MODEL_PATH = f"{script_dir}/jsons/specs/arcturus_#3_left.json"
MODEL_PATH_SAVE = f"{script_dir}/jsons/specs/arcturas_#3_left_new.json"

MODEL_DATA, DIRECTION = init_coord_json(MODEL_PATH)
# MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/arcturas_left_1_self.json"))
# MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/rifts_left_2.json"))
# MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/rifts_right_9.json"))
# MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/semi_slam_plane.json"))
# MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/semi_slam_curve.json"))
# MODEL_DATA, DIRECTION = init_coord_json(os.path.join(script_dir, f"./jsons/specs/semi_slam_polyhedron.json"))

# poly_hedron_cadidates = [
# [-0.06308038, 0.01984174, -0.02024762],
# [-0.04824327, 0.00533525, -0.01977697],
# [-0.01890498, 0.01978277, -0.0051605],
# [-0.00201741, 0.00495996, -0.00489448],
# [0.01838035, 0.02010325, -0.0046952],
# [0.04961416, 0.01984763, -0.0200507],
# [0.06292902, 0.00512941, -0.02017453],
# ]

# curve_candidates = [
# [-0.06459047, 0.02002626, -0.03601638],
# [-0.05198585, 0.0050076, -0.02130529],
# [-0.03051528, 0.01999273, -0.00731907],
# [-0.0104295, 0.00495597, -0.00155286],
# [0.01470543, 0.0200651, -0.00227665],
# [0.03910384, 0.00499392, -0.01158884],
# [0.05206699, 0.01994052, -0.02125366],
# [0.06446175, 0.0050179, -0.03607852],
# ]

# plane_candidates = [
# [-0.0630138, 0.01960507, -0.01325],
# [-0.04843088, 0.00538483, -0.01325],
# [-0.01995668, 0.01994691, -0.01325],
# [-0.0019146, 0.00521891, -0.01325],
# [0.01933924, 0.01994614, -0.01325],
# [0.04776097, 0.00499845, -0.01325],
# [0.06305025, 0.01989964, -0.01325],
# ]


# right
CALIBRATION_DATA_RIGHT = np.array([
[-0.00538706, -0.03670272, 0.00437801],
[0.00970257, -0.04742658, 0.00403234],
[0.02994988, -0.05139695, 0.00403777],
[0.05154067, -0.04528082, 0.0032369],
[0.0694731, -0.02939228, 0.00290863],
[0.07734684, -0.01303514, 0.00307708],
[0.07828127, 0.00929419, 0.0027717],
[0.0717596, 0.02620511, 0.00299559],
[0.05326513, 0.04464458, 0.00333004],
[0.03440088, 0.05093862, 0.0037629],
[0.01250172, 0.04827563, 0.00409494],
[-0.00526364, 0.03655949, 0.00427516],
[-0.0171765, -0.02501771, 0.01713227],
[-0.00734049, -0.03627547, 0.01707953],
[0.0246043, -0.05086197, 0.01805158],
[0.04452817, -0.04733514, 0.01844099],
[0.06129338, -0.03579531, 0.01881552],
[0.07367279, -0.01469268, 0.01948674],
[0.07167918, 0.02059629, 0.01915552],
[0.05464641, 0.04177604, 0.01876967],
[0.03419431, 0.05026223, 0.01837984],
[0.01251094, 0.04860665, 0.01770466],
[-0.00731751, 0.03639945, 0.0171128],
[-0.01727847, 0.02497431, 0.01699061],
])


# left
CALIBRATION_DATA_LEFT = np.array([
[-0.00530853, 0.03670528, 0.00450656],
[0.00991651, 0.04729552, 0.00427008],
[0.02989983, 0.05129122, 0.00389711],
[0.05166774, 0.0455437, 0.00355185],
[0.06968015, 0.02953123, 0.00317441],
[0.0776402, 0.01311999, 0.00257383],
[0.07856579, -0.00967524, 0.00311748],
[0.0718207, -0.02643761, 0.00306791],
[0.05322226, -0.04474321, 0.00329416],
[0.03448044, -0.05097576, 0.00326973],
[0.01253061, -0.04841208, 0.00392918],
[-0.00540531, -0.03663209, 0.00472187],
[-0.01733685, 0.02492997, 0.01676121],
[-0.00754295, 0.03609373, 0.01748052],
[0.02446655, 0.05075625, 0.01762213],
[0.04442015, 0.04720989, 0.01848281],
[0.06135533, 0.03604676, 0.01876286],
[0.07392909, 0.01499913, 0.01918799],
[0.07144154, -0.02046367, 0.01935277],
[0.05430093, -0.04139981, 0.01882305],
[0.03415692, -0.05011215, 0.01816462],
[0.01234603, -0.04867213, 0.01782269],
[-0.00734349, -0.03639581, 0.01731878],
[-0.01731616, -0.02492293, 0.01686719],
])

BLOB_CNT = len(MODEL_DATA)
print('PTS')
for i, leds in enumerate(MODEL_DATA):
    print(f"{np.array2string(leds, separator=', ')},")
print('DIR')
for i, dir in enumerate(DIRECTION):
    print(f"{np.array2string(dir, separator=', ')},")
    
show_calibrate_data(np.array(MODEL_DATA), np.array(DIRECTION), TARGET=np.array(CALIBRATION_DATA_LEFT))


# print('calibrate data save start')
# json_data = rw_json_data(READ, MODEL_PATH, None)
# json_data_cpy = copy.deepcopy(json_data)
# for i, jdata in enumerate(json_data['TrackedObject']['ModelPoints']):
#     json_data_cpy['TrackedObject']['ModelPoints'].get(jdata)[0:3] = CALIBRATION_DATA_LEFT[i]
#     # print(json_data_cpy['TrackedObject']['ModelPoints'].get(jdata))


# rw_json_data(WRITE, MODEL_PATH_SAVE, json_data_cpy)
# print(f"Saved to: \"{MODEL_PATH_SAVE}\".")

