from Advanced_Function import *

'''
1. Model DATA neighbours lists 만들기
    - 각 id에서 내각 90이내 grouping
    - 거리로 sorting
2. 2D Blob Undistort Points
3. 각 2D Blob에서 5개까지 neighbours lists 만들기
    - 거리로 sorting
4. Model DATA와 2D blobs의 조합
5. lambda twist
6. 4번째 led로 checking
7. matching led score 계산
    - 6번에서 만족된 pose로 visible led를 찾음
    - projectPonts하여 몇개나 box안에 들어오는지 check
'''

pebble_camera_matrix = [
    # 0번 sensor
    [np.array([[240.699213, 0.0, 313.735554],
               [0.0, 240.394949, 235.316344],
               [0.0, 0.0, 1.0]], dtype=np.float64),
     np.array([[0.040384], [-0.015174], [-0.000401], [-0.000584]], dtype=np.float64)],
]

MODEL_DATA = np.array([
[-0.03670253, -0.00136743, -0.00683186],
[-0.04727607,  0.01015054,  0.00304395],
[-0.05133603,  0.02573107,  0.01582152],
[-0.0452342 ,  0.04188155,  0.03007053],
[-0.02946931,  0.0556688 ,  0.04175622],
[-0.01309678,  0.06193449,  0.0466701 ],
[0.00930992, 0.0623894 , 0.0474525 ],
[0.02620113, 0.05744795, 0.04307233],
[0.0446104 , 0.04333093, 0.03110736],
[0.05094375, 0.02902339, 0.0188578 ],
[0.04827114, 0.01225686, 0.00472355],
[ 0.03652419, -0.00134508, -0.00666003],
[-0.02510102, -0.00254808, -0.02415428],
[-0.03632259,  0.00511761, -0.01786462],
[-0.05074652,  0.03046963,  0.00159084],
[-0.04730652,  0.04610049,  0.01385638],
[-0.03579857,  0.05931999,  0.02416381],
[-0.01468638,  0.06936861,  0.03150711],
[0.02063129, 0.06767503, 0.03060774],
[0.04178583, 0.05421194, 0.02005855],
[0.05023091, 0.03809911, 0.00740663],
[ 0.03636108,  0.00516153, -0.01790085],
[ 0.02493721, -0.00263817, -0.02406681],
])

DIRECTION = np.array([
[-0.75584205, -0.62542513, -0.19376841],
[-9.0003042e-01, -4.3582690e-01,  3.9198000e-04],
[-0.95576858, -0.0870913 ,  0.28092976],
[-0.84107304,  0.28591898,  0.45918022],
[-0.54169635,  0.61032771,  0.57798369],
[-0.23536262,  0.71232422,  0.66120998],
[0.16887833, 0.72298488, 0.66990519],
[0.48070836, 0.64946134, 0.58916843],
[0.82528168, 0.31942899, 0.46569869],
[ 0.94275266, -0.02286877,  0.33270775],
[ 0.92057199, -0.37444458,  0.11107864],
[ 0.75584205, -0.62542513, -0.19376841],
[-0.64331363, -0.49735623, -0.58205185],
[-0.74783871, -0.40384633, -0.52692068],
[-0.93983665,  0.22402799, -0.25791187],
[-0.8465559 ,  0.52839868, -0.06432689],
[-0.62648053,  0.77498893,  0.08315228],
[-0.25119497,  0.95204283,  0.17468698],
[0.35282319, 0.92066384, 0.16701463],
[0.73368009, 0.67912607, 0.02238979],
[ 0.90365993,  0.38806909, -0.18111078],
[ 0.74783871, -0.40384633, -0.52692068],
[ 0.57732451, -0.53728441, -0.61483484],
])
# points2D_D 초기화
points2D_D = np.array([
    [447.979401, 144.898239],
    [435.574677, 145.552032],
    [411.985107, 146.596695],
    [391.150848, 148.510162],
    [398.024750, 147.767563],
    [450.171570, 154.882889],
    [440.099121, 156.250366],
    [393.340179, 158.017059],
    [414.117188, 157.718704],
    [429.436829, 157.300140],
    [403.095795, 158.119385]
], dtype=np.float64)



def correspondence_search_set_blobs():
    def distance(point1, point2):
        return np.sqrt ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    camera_matrix = pebble_camera_matrix[0][0]
    dist_coeffs = pebble_camera_matrix[0][1]

    points2D_U = cv2.fisheye.undistortPoints ( points2D_D.reshape (-1, 1, 2), camera_matrix, dist_coeffs)
    points2D_U = np.array ( points2D_U.reshape (len(points2D_U), -1), dtype= np.float64 )

    # print(f"points2D_U {points2D_U}")
    sorted_neighbors = []

    for i, anchor in enumerate(points2D_D):
        distances = [(NOT_SET, i, 0.0, anchor, points2D_U[i])] # 현재 anchor 자신을 추가
        distances += [(NOT_SET, j, distance(anchor, point), point, points2D_U[j]) for j, point in enumerate(points2D_D) if i != j]
        sorted_by_distance = sorted(distances, key=lambda x: x[2])
        sorted_neighbors.append (sorted_by_distance[:6]) # anchor 포함해서 6개까지

    print(f"Sorted neighbors by distance")
    for i, neighbors in enumerate(sorted_neighbors):
        print(f"Model 0, blob {i} @ {points2D_D[i][0]:.6f} , {points2D_D[i][1]:.6f} neighbours {len(neighbors)} Search list:")
        for ID, j, dist, point, point_u in neighbors:
            print(f"LED ID {ID} ( {point_u} ) @ {point[0]:.6f} , {point[1]:.6f}. Dist {dist:.6f}")

    return np.array(sorted_neighbors)



class Vec3f:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.arr = np.array ([x, y, z])
    def inverse(self):
        return Vec3f(-self.x, -self.y, -self.z)
    def get_sqlength(self):
        return np.dot ( self.arr , self.arr )
    def ovec3f_get_length(self):
        return math.sqrt (self.x**2 + self.y**2 + self.z**2) 
    def get_dot(self, vec):
        return np.dot ( self.arr , vec.arr )
    def distance_to(self, other_vec):
        return np.linalg.norm ( self.arr - other_vec.arr )
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
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    def normalize_me(self):
        len = self.get_length ()
        self.x /= len
        self.y /= len
        self.z /= len
        self.w /= len
    def get_length(self):
        return math.sqrt (self.x**2 + self.y**2 + self.z**2 + self.w**2)
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
        
def compare_leds(c, l):
    tmp = Vec3f.ovec3f_add(l.pos, c.pose.pos)
    led_pos = Quatf.oquatf_get_rotated(c.pose.orient , tmp)

    return round(math.pow (led_pos.x, 2) + math.pow (led_pos.y, 2), 8)

def calc_led_dist(c, l):
    tmp = Vec3f.ovec3f_add(l.pos, c.pose.pos)
    led_pos = Quatf.oquatf_get_rotated(c.pose.orient , tmp)

    return led_pos, round(math.pow (led_pos.x, 2) + math.pow (led_pos.y, 2), 8)

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
        self.neighbours = sorted( self.neighbours , key=lambda x: compare_leds(self, x))

        # print(f"Have { self.num_neighbours } neighbours for LED { self.led.led_id } ({ self.led.pos.x },{ self.led.pos.y },{ self.led.pos.z }) dir ({ self.led.dir.x },{ self.led.dir.y },{ self.led.dir.z }):")
        # print(f"pose pos [{ self.pose.pos.x },{ self.pose.pos.y },{ self.pose.pos.z }]")
        # print(f"pose orient [{ self.pose.orient.x },{ self.pose.orient.y },{ self.pose.orient.z },{ self.pose.orient.w }]")
        for i in range( self.num_neighbours ):
            cur = self.neighbours [i]
            led_pos, distance = calc_led_dist(self, cur)
            # print(f"LED id { cur.led_id } ({ cur.pos.x },{ cur.pos.y },{ cur.pos.z }) ({ cur.dir.x },{ cur.dir.y },{ cur.dir.z }) -> {led_pos.x} {led_pos.y} @ distance {distance}")

def led_search_candidate_new():
    leds = [Led(i, MODEL_DATA[i], DIRECTION[i]) for i in range(len(MODEL_DATA))]
    # Create RiftLeds object
    rift_leds = RiftLeds(len(leds), leds)
    search_model = []
    for i, led in enumerate( rift_leds.points ):
        led.led_id = i
        c = LedSearchCandidate(led, rift_leds)
        search_model.append(c)
        
    return rift_leds, search_model


def quaternion_from_rotation_matrix(R):
    q = Quatf(0, 0, 0, 0)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt (trace + 1.0)
        q.w = -0.25 / s
        q.x = -(R[2, 1] - R[1, 2]) * s
        q.y = -(R[0, 2] - R[2, 0]) * s
        q.z = -(R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt (1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q.w = -(R[2, 1] - R[1, 2]) / s
            q.x = -0.25 * s
            q.y = -(R[0, 1] + R[1, 0]) / s
            q.z = -(R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt (1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q.w = -(R[0, 2] - R[2, 0]) / s
            q.x = -(R[0, 1] + R[1, 0]) / s
            q.y = -0.25 * s
            q.z = -(R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt (1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q.w = -(R[1, 0] - R[0, 1]) / s
            q.x = -(R[0, 2] + R[2, 0]) / s
            q.y = -(R[1, 2] + R[2, 1]) / s
            q.z = -0.25 * s
    return q

def check_led_match(anchor, candidate_list):
    # 조합은 미리 만들어놔도 될 듯?
    SEARCH_BLOBS_MAP = [
        [0, 1, 2, 3],
        [0, 1, 3, 4],
        [0, 1, 4, 5],
        [0, 2, 3, 4],
        [0, 2, 4, 5],
        [0, 3, 4, 5]
    ]
    print(f"check_led_match: {anchor} {candidate_list}")
    candidate_list.insert(0, anchor)

    POINTS3D_candidates_POS = np.array(MODEL_DATA[list(candidate_list), :], dtype=np.float64)
    POINTS3D_candidates_DIR = np.array(DIRECTION[list(candidate_list), :], dtype=np.float64)
    # print(f"points3D_candidate {POINTS3D_candidates}")
    
    for neighbours_2D in SEARCH_BLOBS:
        for blob_searching in SEARCH_BLOBS_MAP:
            points2D_list = neighbours_2D[list(blob_searching), :]
            # print(f"{blob_searching}")
            # print(f"points2D_list {points2D_list}")
            POINTS2D_candidates = np.array([point[-1] for point in points2D_list], dtype=np.float64)
            blob0 = Vec3f(POINTS2D_candidates[0][0], POINTS2D_candidates[0][1], 1.0)
            checkblob = Vec3f(POINTS2D_candidates[3][0], POINTS2D_candidates[3][1], 1.0)
            
            # 이거 문제임. 다름....
            retval, rvec, tvec = cv2.solveP3P(POINTS3D_candidates_POS[:3], POINTS2D_candidates[:3], default_cameraK, default_dist_coeffs, flags=cv2.SOLVEPNP_P3P)

            for ret_i in range(retval):
                # rvec를 회전 행렬로 변환
                _, rot_matrix = cv2.Rodrigues (rvec[ret_i])

                q = quaternion_from_rotation_matrix(rot_matrix)
                
                pose = Pose(Vec3f(round(tvec[ret_i][0][0], 8),
                                  round(tvec[ret_i][1][0], 8),
                                  round(tvec[ret_i][2][0], 8)), q)
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
                
                print(f"pose.pos {pose.pos.x} {pose.pos.y} {pose.pos.z}")  
                print(f"pose.orient {pose.orient.x} {pose.orient.y} {pose.orient.z} {pose.orient.w}") 
                print(f"blob0 {blob0.x} {blob0.y} {blob0.z}")      
                print(f"checkblob {checkblob.x} {checkblob.y} {checkblob.z}")             
                print(f"checkpos {checkpos}")
                print(f"checkdir {checkdir}")
                
                # return
                
                facing_dot = Vec3f.get_dot(checkpos, checkdir)
                
                if facing_dot > 0:
                    # print("invalid pose")
                    continue           
                
                checkpos = Vec3f.ovec3f_multiply_scalar(checkpos, 1.0/checkpos.z)
                tmp = Vec3f.ovec3f_substract(checkpos, blob0)
                l = Vec3f.ovec3f_get_length(tmp)
                if l > 0.0025:
                    # print(f"Error pose candidate")
                    continue
                    
                # check 4th point                
                led_check_pos = Quatf.oquatf_get_rotated(pose.orient ,
                                                   Vec3f(POINTS3D_candidates_POS[3][0],
                                                         POINTS3D_candidates_POS[3][1],
                                                         POINTS3D_candidates_POS[3][2]))
                led_check_pos = Vec3f.ovec3f_add(led_check_pos, pose.pos)
                led_check_pos = Vec3f.ovec3f_multiply_scalar(led_check_pos, 1.0/led_check_pos.z)
                tmp = Vec3f.ovec3f_substract(led_check_pos, checkblob)
                distance = Vec3f.ovec3f_get_length(tmp)
                print(f"distance {distance}")


def select_k_leds_from_n(anchor, candidate_list):
    k = 3  # anchor 포함해서 총 4개의 LED를 선택
    if len(candidate_list) < k:
        return

    for combo in combinations(candidate_list, k):
        check_led_match(anchor, list(combo))
        # 가운데 2개만 바꿔서 새로운 조합 생성
        if len(combo) >= 3:
            swapped = [combo[0], combo[2], combo[1]]
            check_led_match(anchor, swapped)

def generate_led_match_candidates(neighbors, start=0, depth=3, hopping=3):
    anchor = neighbors[start]

    for _ in range(depth):
        if start + hopping >= len(neighbors):
            break

        candidate_list = neighbors[start + hopping:start + hopping + 3]
        if len(candidate_list) != 3:
            break
        # print(f"candidate_list: {candidate_list}")
        select_k_leds_from_n(anchor, candidate_list)
        start += 1  # 다음 이웃으로 이동

def long_search_python():
    # 예제 사용
    NEIGHBOURS_LIST = []
    for idx, data in enumerate(SEARCH_MODEL):
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
        generate_led_match_candidates(neighbors, start=0, depth=3, hopping=3)
            

if __name__ == "__main__":
    BLOB_CNT = len(MODEL_DATA)
    print('PTS')
    for i, leds in enumerate(MODEL_DATA):
        print(f"{np.array2string(leds, separator=', ')},")
    print('DIR')
    for i, dir in enumerate(DIRECTION):
        print(f"{np.array2string(dir, separator=', ')},")

    # show_calibrate_data(np.array(MODEL_DATA), np.array(DIRECTION))

    RIFT_MODEL, SEARCH_MODEL = led_search_candidate_new()
    SEARCH_BLOBS = correspondence_search_set_blobs()
    long_search_python()











