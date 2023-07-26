from Advanced_Function import *

def camera_displacement(r1, r2, t1, t2):
    print('r1 ', r1)
    print('r2 ', r2)
    Rod1, _ = cv2.Rodrigues(r1)
    Rod2, _ = cv2.Rodrigues(r2)
    R1to2 = Rod2.dot(Rod1.T)
    rvec1to2, _ = cv2.Rodrigues(R1to2)
    tvec1to2 = -R1to2.dot(t1) + t2

    print('Rod1\n', Rod1)
    print('Rod2\n', Rod2)
    print('rvec1to2\n', rvec1to2.T)
    print('tvec1to2\n', tvec1to2.T)

    return rvec1to2, tvec1to2

RVEC_1 =  np.array([-1.42530137, -0.70336427, -1.63589808])
TVEC_1 =  np.array([-0.05003592,  0.00786672, -0.34936662])

RVEC_2 =  np.array([-1.55601754, -0.93174683, -1.58484072])
TVEC_2 =  np.array([-0.0554205,   0.0145185,  -0.38530823])


camera_displacement(RVEC_1, RVEC_2, TVEC_1, TVEC_2)