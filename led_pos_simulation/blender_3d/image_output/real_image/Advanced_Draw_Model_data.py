from Advanced_Function import *

'''
ORIGINAL 3D Points
'''

# MODEL_PATH = f"{script_dir}/jsons/specs/arcturus_#3_right+.json"
# MODEL_PATH_SAVE = f"{script_dir}/jsons/specs/arcturas_#3_right_new.json"

# MODEL_PATH = f"{script_dir}/jsons/specs/arcturus_#3_right_new_cal_sensor_cal_LED21X_ref_out.json"
# MODEL_PATH_SAVE = f"{script_dir}/jsons/specs/rifts_right_new.json"

# MODEL_PATH = f"{script_dir}/jsons/specs/arcturus_#3_left.json"
# MODEL_PATH_SAVE = f"{script_dir}/jsons/specs/arcturas_#3_left_new.json"

# MODEL_PATH = f"{script_dir}/jsons/specs/rifts_left.json"
# MODEL_PATH_SAVE = f"{script_dir}/jsons/specs/rifts_left_new.json"

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
[-0.00538225, -0.03670253, 0.00442442],
[0.00978572, -0.04727607, 0.0040668],
[0.02993561, -0.05133603, 0.00403334],
[0.05145799, -0.0452342, 0.00322126],
[0.06952839, -0.02946931, 0.00289837],
[0.0774895, -0.01309678, 0.00305882],
[0.07833686, 0.00930992, 0.00274091],
[0.07173884, 0.02620113, 0.00300525],
[0.05323637, 0.0446104, 0.00333569],
[0.03440627, 0.05094375, 0.00376592],
[0.01247952, 0.04827114, 0.00409928],
[-0.00525621, 0.03652419, 0.00430552],
[-0.01725874, -0.02510102, 0.0170897],
[-0.00734298, -0.03632259, 0.01707089],
[0.02459884, -0.05074652, 0.01805065],
[0.04446367, -0.04730652, 0.01844551],
[0.06122225, -0.03579857, 0.01883048],
[0.07364986, -0.01468638, 0.01950387],
[0.07176939, 0.02063129, 0.01912847],
[0.0546692, 0.04178583, 0.01877653],
[0.03418668, 0.05023091, 0.01837584],
[0.01247775, 0.0485895, 0.01770159],
[-0.0073319, 0.03636108, 0.01712673],
[-0.01727314, 0.02493721, 0.01696496],
])


# left
CALIBRATION_DATA_LEFT = np.array([
[-0.00529981, 0.03665592, 0.00450667],
[0.00987789, 0.04735335, 0.00422782],
[0.02992446, 0.05135677, 0.00389418],
[0.05172681, 0.04554052, 0.00346661],
[0.06965931, 0.0294462, 0.0030962],
[0.07751442, 0.01305547, 0.00251284],
[0.07848266, -0.00963751, 0.00314491],
[0.07181742, -0.02641001, 0.00314152],
[0.05312819, -0.0445685, 0.00328151],
[0.03446891, -0.05088316, 0.00327581],
[0.01255621, -0.04854797, 0.00395561],
[-0.00536169, -0.03671202, 0.0047295],
[-0.01727304, 0.02492984, 0.01687438],
[-0.00749085, 0.03614692, 0.01731315],
[0.02446019, 0.05077714, 0.01766438],
[0.04444107, 0.04714564, 0.0186535],
[0.06136127, 0.03595401, 0.01873972],
[0.0738646, 0.01489609, 0.01925076],
[0.0714089, -0.02044542, 0.01941797],
[0.05420588, -0.04122241, 0.01878042],
[0.03413652, -0.05003679, 0.01810631],
[0.01238486, -0.04881236, 0.01777241],
[-0.00726006, -0.03644273, 0.01734367],
[-0.01714664, -0.02485884, 0.01687096],
])

BLOB_CNT = len(MODEL_DATA)
print('PTS')
for i, leds in enumerate(MODEL_DATA):
    print(f"{np.array2string(leds, separator=', ')},")
print('DIR')
for i, dir in enumerate(DIRECTION):
    print(f"{np.array2string(dir, separator=', ')},")


print(f"len(MODEL_DATA) {len(MODEL_DATA)} len(CALIBRATION_DATA_RIGHT) {len(CALIBRATION_DATA_RIGHT)}")
show_calibrate_data(np.array(MODEL_DATA), np.array(DIRECTION), TARGET=np.array(CALIBRATION_DATA_RIGHT))


print('calibrate data save start')
json_data = rw_json_data(READ, MODEL_PATH, None)
json_data_cpy = copy.deepcopy(json_data)
for i, jdata in enumerate(json_data['TrackedObject']['ModelPoints']):
    json_data_cpy['TrackedObject']['ModelPoints'].get(jdata)[0:3] = CALIBRATION_DATA_RIGHT[i]
    # print(json_data_cpy['TrackedObject']['ModelPoints'].get(jdata))


rw_json_data(WRITE, MODEL_PATH_SAVE, json_data_cpy)
print(f"Saved to: \"{MODEL_PATH_SAVE}\".")

