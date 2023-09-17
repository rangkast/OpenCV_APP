from Advanced_Function import *

class Vec3f:
    def __init__(self, x, y, z):
        self.x = round(x, 8)
        self.y = round(y, 8)
        self.z = round(z, 8)
        self.arr = np.array ([x, y, z])
    def inverse(self):
        return Vec3f(-self.x, -self.y, -self.z)
    def get_sqlength(self):
        return round(np.dot ( self.arr , self.arr ), 8)
    def ovec3f_get_length(self):
        return round(math.sqrt (self.x**2 + self.y**2 + self.z**2), 8)
    def get_dot(self, vec):
        return np.dot ( self.arr , vec.arr )
    def distance_to(self, other_vec):
        return round(np.linalg.norm ( self.arr - other_vec.arr ), 8)
    def ovec3f_add(a, b):
        return Vec3f(a.x + b.x, a.y + b.y, a.z + b.z)
    def ovec3f_substract(a, b):
        return Vec3f(a.x - b.x, a.y - b.y, a.z - b.z)
    def ovec3f_multiply_scalar(a, S):
        return Vec3f(a.x*S, a.y*S, a.z*S)
    def ovec3f_normalize_me(self):
        if self.x == 0 and self.y == 0 and self.z == 0:
            return
        len = np.linalg.norm ( self.arr )
        self.x /= len
        self.y /= len
        self.z /= len
        return Vec3f(self.x, self.y, self.z) # Update the 'arr' attribute as well
    def __repr__(self):
        return f"({self.x} {self.y} {self.z})"
class Quatf:
    def __init__(self, x, y, z, w):
        self.x = round(x, 8)
        self.y = round(y, 8)
        self.z = round(z, 8)
        self.w = round(w, 8)
    def normalize_me(self):
        len = self.get_length ()
        self.x /= len
        self.y /= len
        self.z /= len
        self.w /= len
    def get_length(self):
        return round(math.sqrt (self.x**2 + self.y**2 + self.z**2 + self.w**2), 8)
    # Oquatf get_rotated
    def oquatf_get_rotated(me, vec):
        q = Quatf(vec.x * me.w + vec.z * me.y - vec.y * me.z,
                    vec.y * me.w + vec.x * me.z - vec.z * me.x,
                    vec.z * me.w + vec.y * me.x - vec.x * me.y,
                    vec.x * me.x + vec.y * me.y + vec.z * me.z)
        x = round(me.w * q.x + me.x * q.w + me.y * q.z - me.z * q.y, 8)
        y = round(me.w * q.y + me.y * q.w + me.z * q.x - me.x * q.z, 8)
        z = round(me.w * q.z + me.z * q.w + me.x * q.y - me.y * q.x, 8)
        return Vec3f(x, y, z)
                       
    @staticmethod
    def from_vectors(from_vec, to_vec):
        axis = np.cross ( from_vec.arr , to_vec.arr )
        dot = from_vec.get_dot (to_vec)
        if np.linalg.norm (axis)**2 < 1e-7:
            if dot < 0:
                return Quatf(from_vec.x, -from_vec.z, from_vec.y, 0).normalize_me()
            else:
                return Quatf(0, 0, 0, 1)
        else:
            w = dot + math.sqrt ( from_vec.get_sqlength () * to_vec.get_sqlength ())
            q = Quatf(axis[0], axis[1], axis[2], w)
            q.normalize_me ()
            return q
class Led:
    def __init__(self, led_id, pos, dir):
        self.led_id = led_id
        self.pos = Vec3f(pos[0], pos[1], pos[2])
        self.dir = Vec3f(dir[0], dir[1], dir[2])
class Pose:
    def __init__(self, pos, orient):
        self.pos = pos
        self.orient = orient
class RiftLeds:
    def __init__(self, num_points, points):
        self.num_points = num_points
        self.points = points
class LedSearchCandidate:
    def __init__(self, led, led_model):
        self.led = led
        self.neighbours = []
        self.num_neighbours = 0
        self.pose = Pose(Vec3f.inverse(self.led.pos), Quatf.from_vectors ( self.led.dir , Vec3f(0.0, 0.0, 1.0)))

        for i in range( led_model.num_points ):
            cur = led_model.points [i]
            if cur == self.led :
                continue
            if self.led.dir.get_dot ( cur.dir ) <= 0:
                continue
            self.neighbours.append (cur)
            self.num_neighbours += 1
        # Sort neighbours based on the projected distance
        self.neighbours = sorted( self.neighbours , key=lambda x: self.compare_leds(x))

        # print(f"Have { self.num_neighbours } neighbours for LED { self.led.led_id } ({ self.led.pos.x },{ self.led.pos.y },{ self.led.pos.z }) dir ({ self.led.dir.x },{ self.led.dir.y },{ self.led.dir.z }):")
        # print(f"pose pos [{ self.pose.pos.x },{ self.pose.pos.y },{ self.pose.pos.z }]")
        # print(f"pose orient [{ self.pose.orient.x },{ self.pose.orient.y },{ self.pose.orient.z },{ self.pose.orient.w }]")
        for i in range( self.num_neighbours ):
            cur = self.neighbours [i]
            led_pos, distance = self.calc_led_dist(cur)
            # print(f"LED id { cur.led_id } ({ cur.pos.x },{ cur.pos.y },{ cur.pos.z }) ({ cur.dir.x },{ cur.dir.y },{ cur.dir.z }) -> {led_pos.x} {led_pos.y} @ distance {distance}")
    
    def compare_leds(self, l):
            tmp = Vec3f.ovec3f_add(l.pos, self.pose.pos)
            led_pos = Quatf.oquatf_get_rotated(self.pose.orient, tmp)

            return round(math.pow(led_pos.x, 2) + math.pow(led_pos.y, 2), 8)

    def calc_led_dist(self, l):
        tmp = Vec3f.ovec3f_add(l.pos, self.pose.pos)
        led_pos = Quatf.oquatf_get_rotated(self.pose.orient, tmp)

        return led_pos, round(math.pow(led_pos.x, 2) + math.pow(led_pos.y, 2), 8)   
    
# 라이브러리 로드
LAMBDATWIST_LIB = cdll.LoadLibrary(f"{script_dir}/../../../../EXTERNALS/lambdatwist_p3p.so")

lambdatwist_p3p = LAMBDATWIST_LIB.lambdatwist_p3p
lambdatwist_p3p.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64),  # iny1
    np.ctypeslib.ndpointer(dtype=np.float64),  # iny2
    np.ctypeslib.ndpointer(dtype=np.float64),  # iny3
    np.ctypeslib.ndpointer(dtype=np.float64),  # x1
    np.ctypeslib.ndpointer(dtype=np.float64),  # x2
    np.ctypeslib.ndpointer(dtype=np.float64),  # x3
    np.ctypeslib.ndpointer(dtype=np.float64, shape=(4,9)),  # R
    np.ctypeslib.ndpointer(dtype=np.float64, shape=(4,3))   # t
]

DEG_TO_RAD = pi / 180.0
RIFT_LED_ANGLE = 75  # This value should be set to whatever the real angle is in your use case
MATCH_STRONG = 2
MATCH_GOOD = 1
def OHMD_MAX(_a, _b):
    return _a if _a > _b else _b
def LED_OBJECT_ID(l):
    return l if l < 0 else l >> 8
def find_best_matching_led(led_points, num_leds, blob):
    best_z = float('inf')
    best_led_index = -1
    best_sqerror = 1e20
    leds_within_range = 0

    for i in range(num_leds):
        led_info = led_points[i]
        pos_px = led_info['pos_px']
        led_radius_px = led_info['led_radius_px']

        dx = abs(pos_px[0] - blob[0])
        dy = abs(pos_px[1] - blob[1])

        sqerror = dx ** 2 + dy ** 2

        if sqerror < (led_radius_px ** 2):
            leds_within_range += 1

            if best_led_index < 0 or best_z > led_info['pos_m'].z or (sqerror + led_radius_px) < best_sqerror:
                best_z = led_info['pos_m'].z
                best_led_index = i
                best_sqerror = sqerror

    return best_led_index, best_sqerror

def quaternion_from_rotation_matrix(R):
    trace = R[0] + R[4] + R[8]
    if trace > 0:
        s = 0.5 / math.sqrt (trace + 1.0)
        w = -0.25 / s
        x = -(R[7] - R[5]) * s
        y = -(R[2] - R[6]) * s
        z = -(R[3] - R[1]) * s
    else:
        if R[0] > R[4] and R[0] > R[8]:
            s = 2.0 * math.sqrt (1.0 + R[0] - R[4] - R[8])
            w = -(R[7] - R[5]) / s
            x = -0.25 * s
            y = -(R[1] + R[3]) / s
            z = -(R[2] + R[6]) / s
        elif R[4] > R[8]:
            s = 2.0 * math.sqrt (1.0 + R[4] - R[0] - R[8])
            w = -(R[2] - R[6]) / s
            x = -(R[1] + R[3]) / s
            y = -0.25 * s
            z = -(R[5] + R[7]) / s
        else:
            s = 2.0 * math.sqrt (1.0 + R[8] - R[0] - R[4])
            w = -(R[3] - R[1]) / s
            x = -(R[2] + R[6]) / s
            y = -(R[5] + R[7]) / s
            z = -0.25 * s
    return Quatf(x, y, z, w)
     
def quaternion_to_rotvec(quaternion: Quatf):
    qw, qx, qy, qz = quaternion.w, quaternion.x, quaternion.y, quaternion.z
    angle = 2 * np.arccos(qw)
    sin_half_angle = np.sqrt(1.0 - qw * qw)
    
    if sin_half_angle == 0:
        return np.array([0.0, 0.0, 0.0])
    
    rx = qx / sin_half_angle
    ry = qy / sin_half_angle
    rz = qz / sin_half_angle
    
    rvec = angle * np.array([rx, ry, rz])
    
    return rvec

def led_search_candidate_new(MODEL_DATA, DIRECTION):
    leds = [Led(i, MODEL_DATA[i], DIRECTION[i]) for i in range(len(MODEL_DATA))]
    # Create RiftLeds object
    rift_leds = RiftLeds(len(leds), leds)
    search_model = []
    for i, led in enumerate( rift_leds.points ):
        led.led_id = i
        c = LedSearchCandidate(led, rift_leds)
        search_model.append(c)
        
    return rift_leds, search_model


def correspondence_search_set_blobs(points2D_D, camera_m, blobs=[]):
    def distance(point1, point2):
        return np.sqrt ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    # print(f"points2D_U {points2D_U}")
    points2D_U = cv2.fisheye.undistortPoints (points2D_D.reshape (-1, 1, 2), 
                                                camera_m[0][0],
                                                camera_m[0][1])
    points2D_U = np.array ( points2D_U.reshape (len(points2D_U), -1), dtype= np.float64 )
    sorted_neighbors = []
    # print(f"camera_m\n{camera_m[0][0][1]}")
    max_size_array = []
    size_calc = 0

    if len(blobs) > 0:
        size_calc = 1
        for blob in blobs:
            max_size = np.max([blob[4][2]/camera_m[0][0][0][0], blob[4][3]/camera_m[0][0][1][1]])
            max_size_array.append(max_size)    
        # print(f"max_size_array {max_size_array} len {len(max_size_array)}")

    for i, anchor in enumerate(points2D_D):
        distances = [(NOT_SET,  i, 0.0, anchor, points2D_U[i], blobs[i] if size_calc == 1 else 0, max_size_array[i] if size_calc == 1 else 0)] # 현재 anchor 자신을 추가
        distances += [(NOT_SET, j, distance(anchor, point), point, points2D_U[j], blobs[j] if size_calc == 1 else 0,  max_size_array[j] if size_calc == 1 else 0) for j, point in enumerate(points2D_D) if i != j]
        sorted_by_distance = sorted(distances, key=lambda x: x[2])
        sorted_neighbors.append (sorted_by_distance[:6])

    print(f"Sorted neighbors by distance")
    for i, neighbors in enumerate(sorted_neighbors):
        print(f"Model 0, blob {i} @ {points2D_D[i][0]:.6f} , {points2D_D[i][1]:.6f} neighbours {len(neighbors)} Search list:")
        for ID, j, dist, point, point_u, blobs_detected, max_sized in neighbors:
            print(f"LED ID {ID} ( {point_u} ) @ {point[0]:.6f} , {point[1]:.6f}. Dist {dist:.6f} blobs {blobs_detected} size {max_sized}")

    return np.array(sorted_neighbors, dtype=object), points2D_U


def rift_evaluate_pose_with_prior(DATA_SET, pose, pose_prior=None):
    led_radius_mm = 2.0 / 1000.0
    first_visible = True
    MODEL_DATA = DATA_SET[0]
    DIRECTION = DATA_SET[1]
    camera_m = DATA_SET[4]
    points2D_D = DATA_SET[5]
    score = DATA_SET[7]

    BLOB_CNT = len(MODEL_DATA)
    # print("MODEL_DATA dtype:", MODEL_DATA.dtype)
    # print("MODEL_DATA shape:", MODEL_DATA.shape)
    # Extract quaternion from the pose
    quaternion = Quatf(pose.orient.x, pose.orient.y, pose.orient.z, pose.orient.w)

    # Unpack the pose and camera matrix
    tvec = np.array([pose.pos.x, pose.pos.y, pose.pos.z], dtype=np.float64)
    # Convert the quaternion to rotation vector
    rvec = np.array(quaternion_to_rotvec(quaternion), dtype=np.float64)

    # Project the 3D LED points onto the 2D image plane
    projected_points, _ = cv2.fisheye.projectPoints(MODEL_DATA.reshape (-1, 1, 3), rvec, tvec,
                                                    camera_m[0][0],
                                                    camera_m[0][1])
    flattened_points = projected_points.squeeze()
    # print(f"projected_points\n{flattened_points}")
     # Get the focal length from the camera matrix
    focal_length = OHMD_MAX(camera_m[0][0][0, 0], camera_m[0][0][1, 1])
    
    # Initialize variables
    visible_led_points = []
    bounds = {"left": float('inf'), "top": float('inf'), "right": float('-inf'), "bottom": float('-inf')}

    # Iterate through the LED points (assuming they are in MODEL_DATA)
    num_leds = BLOB_CNT
    for i in range(num_leds):
        led_pos_px = flattened_points[i]
        
        # Check if LED is within screen bounds
        if led_pos_px[0] < 0 or led_pos_px[1] < 0 or led_pos_px[0] >= camera_m[0][0][0, 2] * 2 or led_pos_px[1] >= camera_m[0][0][1, 2] * 2:
            continue  # Skip this iteration
        
        # print(f"{Vec3f(MODEL_DATA[i][0], MODEL_DATA[i][1], MODEL_DATA[i][2])}")
        # Create instances of your custom Vec3f class
        led_pos_m = Quatf.oquatf_get_rotated(pose.orient, Vec3f(MODEL_DATA[i][0], MODEL_DATA[i][1], MODEL_DATA[i][2]))
        led_pos_m = Vec3f.ovec3f_add(pose.pos, led_pos_m)   

        # Calculate LED radius in pixels
        led_radius_px = 4.0
        if led_pos_m.z > 0:
            led_radius_px = focal_length * led_radius_mm / led_pos_m.z  # led_radius_mm needs to be defined

        tmp = copy.deepcopy(led_pos_m)
        tmp = Vec3f.ovec3f_normalize_me(tmp)
        normal = Quatf.oquatf_get_rotated(pose.orient , Vec3f(DIRECTION[i][0], DIRECTION[i][1], DIRECTION[i][2]))          
        facing_dot = Vec3f.get_dot(tmp, normal)
        # print(f"LED {i} pos {led_pos_m.x},{led_pos_m.y},{led_pos_m.z} -> {led_pos_px[0]},{led_pos_px[1]}  facing_dot {facing_dot}")
        # Check the facing_dot value against a threshold
        if facing_dot < cos(DEG_TO_RAD * (180.0 - RIFT_LED_ANGLE)):
            # Append to visible LED points
            visible_led_points.append({
                'pos_px': led_pos_px,
                'pos_m': led_pos_m,
                'led_radius_px': led_radius_px
            })

            # Update bounding box
            if first_visible:
                bounds['left'] = led_pos_px[0] - led_radius_px
                bounds['top'] = led_pos_px[1] - led_radius_px
                bounds['right'] = led_pos_px[0] + led_radius_px
                bounds['bottom'] = led_pos_px[1] + led_radius_px
                first_visible = False
            else:
                bounds['left'] = min(bounds['left'], led_pos_px[0] - led_radius_px)
                bounds['top'] = min(bounds['top'], led_pos_px[1] - led_radius_px)
                bounds['right'] = max(bounds['right'], led_pos_px[0] + led_radius_px)
                bounds['bottom'] = max(bounds['bottom'], led_pos_px[1] + led_radius_px)
    
    score_visible_leds = len(visible_led_points)
    # print(f"score_visible_leds {score_visible_leds}")
    if score_visible_leds < 5:
        score['flags'] = ERROR
        return ERROR, bounds
    
    # print("points2D_D")
    # for blobs in points2D_D:
    #     print(blobs)
    tmp_score = {'matched_id':[], 'matched_blobs': 0, 'unmatched_blobs': 0, 'reprojection_error': 0.0, 'error_per_led': 0}

    for i, blob in enumerate(points2D_D):  # Assuming points2D_D is similar to blobs in C
        # led_object_id = LED_OBJECT_ID(blob['led_id'])  # Assuming blob has 'led_id'        
        # # Skip blobs which already have an ID not belonging to this device
        # if led_object_id != LED_INVALID_ID and led_object_id != device_id:
        #     continue        
        # Check if blob is within the bounding box
        if blob[0] >= bounds['left'] and blob[1] >= bounds['top'] and blob[0] < bounds['right'] and blob[1] < bounds['bottom']:
            sqerror = 0.0  # You might want to replace this with an actual calculation
            match_result = find_best_matching_led(visible_led_points, score_visible_leds, blob)
            if match_result is not None:  # or some other condition to check for valid result
                match_led_index, sqerror = match_result
                if match_led_index >= 0:
                    tmp_score['reprojection_error'] += sqerror
                    tmp_score['matched_blobs'] += 1
                    tmp_score['matched_id'].append([i, match_led_index])
                else:
                    tmp_score['unmatched_blobs'] += 1

    # print(f"score {score}")

    # Check if there are enough matched blobs
    if tmp_score['matched_blobs'] < 5:
        # print("Not enough matched blobs, exiting.")
        score['flags'] = ERROR
        return ERROR, tmp_score
    else:
        error_per_led = tmp_score['reprojection_error'] / tmp_score['matched_blobs']        
        if error_per_led < score['error_per_led'] and tmp_score['reprojection_error'] < score['reprojection_error']:
            score['error_per_led'] = error_per_led        
            score['reprojection_error'] = tmp_score['reprojection_error']
            score['matched_blobs'] = tmp_score['matched_blobs']
            score['unmatched_blobs'] = tmp_score['unmatched_blobs']
            score['matched_id'] = tmp_score['matched_id']
        # print(f"Error per LED: {error_per_led}")
        if error_per_led < 1.5:
            # print(f"MATCH STRONG")
            score['flags'] =  MATCH_STRONG
            return MATCH_STRONG, tmp_score
    score['flags'] = MATCH_GOOD
    return MATCH_GOOD, tmp_score
        

def check_led_match(DATA_SET, anchor, candidate_list):
    # 조합은 미리 만들어놔도 될 듯?
    SEARCH_BLOBS_MAP = [
        [0, 1, 2, 3],
        [0, 1, 3, 4],
        [0, 1, 4, 5],
        [0, 2, 3, 4],
        [0, 2, 4, 5],
        [0, 3, 4, 5]
    ]
    MODEL_DATA = DATA_SET[0]
    DIRECTION = DATA_SET[1]
    SEARCH_BLOBS = DATA_SET[3]

    # print(f"check_led_match: {anchor} {candidate_list}")
    candidate_list.insert(0, anchor)

    POINTS3D_candidates_POS = np.array(MODEL_DATA[list(candidate_list), :], dtype=np.float64)
    POINTS3D_candidates_DIR = np.array(DIRECTION[list(candidate_list), :], dtype=np.float64)
    # print(f"points3D_candidate {POINTS3D_candidates}")

    for neighbours_2D in SEARCH_BLOBS:
        for blob_searching in SEARCH_BLOBS_MAP:
            if DATA_SET[7]['flags'] == MATCH_STRONG:
                return
            points2D_list = neighbours_2D[list(blob_searching), :]
            # print(f"{blob_searching}")
            # print(f"points2D_list {points2D_list}")
            # print(f"neighbours_2D {neighbours_2D}")
            POINTS2D_candidates = np.array([(point[4], point[6]) for point in points2D_list], dtype=object)
            # print(f"POINTS2D_candidates {POINTS2D_candidates}")
            blob0 = Vec3f(POINTS2D_candidates[0][0][0], POINTS2D_candidates[0][0][1], 1.0)
            checkblob = Vec3f(POINTS2D_candidates[3][0][0], POINTS2D_candidates[3][0][1], 1.0)
            checkblob_size = POINTS2D_candidates[3][1]
            # print(f"blob0 {blob0.x} {blob0.y} {blob0.z}")      
            # print(f"checkblob {checkblob.x} {checkblob.y} {checkblob.z} {checkblob_size}")

            # 첫 세 개의 3D 및 2D 포인트
            y1, y2, y3 = [np.append(point[0], 1.0) for point in POINTS2D_candidates[:3]]
            x1, x2, x3 = POINTS3D_candidates_POS[:3]

            # print(f"POINTS2D_candidates")
            # print(f"{y1}\n{y2}\n{y3}")
            # print(f"POINTS3D_candidates_POS")
            # print(f"{x1}\n{x2}\n{x3}")

            R = np.zeros((4, 9), dtype=np.float64)  # 4개의 3x3 행렬
            t = np.zeros((4, 3), dtype=np.float64)  # 4개의 3D 이동 벡터

            valid = lambdatwist_p3p(y1, y2, y3, x1, x2, x3, R, t)

            # print("Rotation matrices:", R)
            # print("Translation vectors:", t)
            # print(f"valid {valid}")

            if valid:
                for i in range(valid):
                    q = quaternion_from_rotation_matrix(R[i])
                    pose = Pose(Vec3f(round(t[i][0], 8),
                                      round(t[i][1], 8),
                                      round(t[i][2], 8)), q)
                    if pose.pos.z < 0.05 or pose.pos.z > 15:
                        continue
                    pose.orient.normalize_me()
                    checkpos = Quatf.oquatf_get_rotated(pose.orient ,
                                                   Vec3f(POINTS3D_candidates_POS[0][0],
                                                         POINTS3D_candidates_POS[0][1],
                                                         POINTS3D_candidates_POS[0][2]))
                    checkpos = Vec3f.ovec3f_add(checkpos, pose.pos)                    
                    checkdir = Quatf.oquatf_get_rotated(pose.orient ,
                                                    Vec3f(POINTS3D_candidates_DIR[0][0],
                                                          POINTS3D_candidates_DIR[0][1],
                                                          POINTS3D_candidates_DIR[0][2]))
                    checkpos = Vec3f.ovec3f_normalize_me(checkpos)
                    facing_dot = Vec3f.get_dot(checkpos, checkdir)

                    # print(f"checkpos {checkpos}")
                    # print(f"checkdir {checkdir}")
                    if facing_dot > 0:
                        # print("invalid pose")
                        continue           
                    
                    checkpos = Vec3f.ovec3f_multiply_scalar(checkpos, 1.0/checkpos.z)
                    tmp = Vec3f.ovec3f_substract(checkpos, blob0)
                    l = Vec3f.ovec3f_get_length(tmp)
                    if l > 0.0025:
                        # print(f"Error pose candidate")
                        continue
                    led_check_pos = Quatf.oquatf_get_rotated(pose.orient ,
                                                    Vec3f(POINTS3D_candidates_POS[3][0],
                                                          POINTS3D_candidates_POS[3][1],
                                                          POINTS3D_candidates_POS[3][2]))

                    led_check_pos = Vec3f.ovec3f_add(led_check_pos, pose.pos)
                    led_check_pos = Vec3f.ovec3f_multiply_scalar(led_check_pos, 1.0/led_check_pos.z)
                    tmp = Vec3f.ovec3f_substract(led_check_pos, checkblob)
                    distance = Vec3f.ovec3f_get_length(tmp)
                    
                    # print(f"ANCHOR [{blob0.x},{blob0.y},{blob0.z}]")
                    # print(f"pose.pos {pose.pos.x} {pose.pos.y} {pose.pos.z}")  
                    # print(f"pose.orient {pose.orient.x} {pose.orient.y} {pose.orient.z} {pose.orient.w}")      
                    # print(f"distance {distance}")

                    # ToDo
                    # 4th blob의 max_size와 distance 비교 추가필요                    
                    if distance <= 0.020 if checkblob_size == 0 else checkblob_size:
                        DATA_SET[7]['flags'] = NOT_SET
                        _, _ = rift_evaluate_pose_with_prior(DATA_SET, pose)
                        if DATA_SET[7]['flags'] == MATCH_STRONG:
                            print(f"MATCH_STRONG score input {DATA_SET[7]}")
                            # print("MATCH_STRONG")
                    # sys.exit()    
                    

def select_k_leds_from_n(DATA_SET, anchor, candidate_list):
    k = 3  # anchor 포함해서 총 4개의 LED를 선택
    if len(candidate_list) < k:
        return

    for combo in combinations(candidate_list, k):
        check_led_match(DATA_SET, anchor, list(combo))
        # 가운데 2개만 바꿔서 새로운 조합 생성
        if len(combo) >= 3:
            if DATA_SET[7]['flags'] == MATCH_STRONG:
                return
            swapped = [combo[0], combo[2], combo[1]]
            check_led_match(DATA_SET, anchor, swapped)

def generate_led_match_candidates(DATA_SET, neighbors, start=0, depth=3, hopping=3):
    anchor = neighbors[start]
    for _ in range(depth):
        if start + hopping >= len(neighbors):
            break
        candidate_list = neighbors[start + hopping:start + hopping + 3]
        if len(candidate_list) != 3:
            break
        # print(f"candidate_list: {candidate_list}")
        select_k_leds_from_n(DATA_SET, anchor, candidate_list)
        start += 1  # 다음 이웃으로 이동

def long_search_python(DATA_SET):
    # 예제 사용
    NEIGHBOURS_LIST = []
    for idx, data in enumerate(DATA_SET[2]):
        anchor_point = data.led.led_id
        # print(f"Anchor {anchor_point}")
        neighbours_list = []
        neighbours_list.append(anchor_point)
        for led_num in data.neighbours:
            # print(led_num.led_id)
            neighbours_list.append(led_num.led_id)
        NEIGHBOURS_LIST.append(neighbours_list) 
    print("NEIGHBOURS_LIST")
    for neighbors in NEIGHBOURS_LIST:
        print(f"neighbours: {neighbors}")
        generate_led_match_candidates(DATA_SET, neighbors, start=0, depth=3, hopping=3)