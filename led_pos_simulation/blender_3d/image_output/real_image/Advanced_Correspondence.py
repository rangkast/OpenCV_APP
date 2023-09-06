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

    print(f"points2D_U {points2D_U}")
    sorted_neighbors = []

    for i, anchor in enumerate(points2D_D):
        distances = [(NOT_SET, j, distance(anchor, point), point) for j, point in enumerate(points2D_D) if i != j]
        sorted_by_distance = sorted(distances, key=lambda x: x[2])
        sorted_neighbors.append(sorted_by_distance[:5])

        print(f"Sorted neighbors by distance")

    for i, neighbors in enumerate(sorted_neighbors):
        print(f"Model 0, blob {i} @ {points2D_D[i][0]:.6f} , {points2D_D[i][1]:.6f} neighbours {len(neighbors)} Search list:")
        for ID, j, dist, point in neighbors:
            print(f"LED ID {ID} ( {points2D_U[j][0]:.6f} , {points2D_U[j][1]:.6f} ) @ {point[0]:.6f} , {point[1]:.6f}. Dist {dist:.6f}")

    return points2D_U, sorted_neighbors



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
    def get_dot(self, vec):
        return np.dot ( self.arr , vec.arr )
    def distance_to(self, other_vec):
        return np.linalg.norm ( self.arr - other_vec.arr )
    
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
        

def ovec3f_add(a, b):
    return Vec3f(a.x + b.x, a.y + b.y, a.z + b.z)

# Oquatf get_rotated
def oquatf_get_rotated(me, vec):
    q = Quatf(vec.x * me.w + vec.z * me.y - vec.y * me.z,
                vec.y * me.w + vec.x * me.z - vec.z * me.x,
                vec.z * me.w + vec.y * me.x - vec.x * me.y,
                vec.x * me.x + vec.y * me.y + vec.z * me.z)
    x = me.w * q.x + me.x * q.w + me.y * q.z - me.z * q.y
    y = me.w * q.y + me.y * q.w + me.z * q.x - me.x * q.z
    z = me.w * q.z + me.z * q.w + me.x * q.y - me.y * q.x

    return Vec3f(x, y, z)

def compare_leds(c, l):
    tmp = ovec3f_add(l.pos, c.pose.pos)
    led_pos = oquatf_get_rotated(c.pose.orient , tmp)

    return math.pow (led_pos.x, 2) + math.pow (led_pos.y, 2)

def calc_led_dist(c, l):
    tmp = ovec3f_add(l.pos, c.pose.pos)
    led_pos = oquatf_get_rotated(c.pose.orient , tmp)

    return led_pos, math.pow (led_pos.x, 2) + math.pow (led_pos.y, 2)

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

        print(f"Have { self.num_neighbours } neighbours for LED { self.led.led_id } ({ self.led.pos.x },{ self.led.pos.y },{ self.led.pos.z }) dir ({ self.led.dir.x },{ self.led.dir.y },{ self.led.dir.z }):")
        print(f"pose pos [{ self.pose.pos.x },{ self.pose.pos.y },{ self.pose.pos.z }]")
        print(f"pose orient [{ self.pose.orient.x },{ self.pose.orient.y },{ self.pose.orient.z },{ self.pose.orient.w }]")
        for i in range( self.num_neighbours ):
            cur = self.neighbours [i]
            led_pos, distance = calc_led_dist(self, cur)
            print(f"LED id { cur.led_id } ({ cur.pos.x },{ cur.pos.y },{ cur.pos.z }) ({ cur.dir.x },{ cur.dir.y },{ cur.dir.z }) -> {led_pos.x} {led_pos.y} @ distance {distance}")

def led_search_candidate_new(led, led_model, led_num):
    led.led_id = led_num
    c = LedSearchCandidate(led, led_model)

    return c


if __name__ == "__main__":
    BLOB_CNT = len(MODEL_DATA)
    print('PTS')
    for i, leds in enumerate(MODEL_DATA):
        print(f"{np.array2string(leds, separator=', ')},")
    print('DIR')
    for i, dir in enumerate(DIRECTION):
        print(f"{np.array2string(dir, separator=', ')},")

    # show_calibrate_data(np.array(MODEL_DATA), np.array(DIRECTION))
    # correspondence_search_set_blobs()
    leds = [Led(i, MODEL_DATA[i], DIRECTION[i]) for i in range(len(MODEL_DATA))]
    # Create RiftLeds object
    rift_leds = RiftLeds(len(leds), leds)

    for i, led in enumerate( rift_leds.points ):
        led_search_candidate_new(led, rift_leds, i)

    









