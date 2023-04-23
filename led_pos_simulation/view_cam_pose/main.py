import numpy as np
import math
object_points = np.array([
    [-0.02160651, -0.00342903, -0.01387145],
    [-0.03193206, 0.00572255, -0.0121337 ],
    [-0.03707899, 0.00930854, 0.00319688],
    [-0.04286609, 0.02717354, -0.00196727],
    [-0.04170636, 0.03584988, 0.01993165],
    [-0.02913154, 0.06187207, 0.01626444],
    [-0.01448837, 0.06300491, 0.03665803],
    [0.00756746, 0.07114425, 0.02058987],
    [0.0299406, 0.05522555, 0.03115318],
    [0.03710101, 0.05282076, 0.01101663],
    [0.04270353, 0.03023362, 0.01622256],
    [0.04221143, 0.02285514, -0.00396752],
    [0.03304973, 0.00356362, 0.0002965 ],
    [0.0302229, 0.0037782, -0.01307012],
    [0.02016397, -0.00415014, -0.01488726]
])


rvec_array = np.array([
    [ 0.84934492, -0.55056834,  0.36718911],
    [ 0.72841522, -1.01090452,  0.70232459],
    [ 0.80816327,  1.60183358, -0.83176954],
    [ 0.57564109,  1.87415192, -1.07864405],
    [ 0.94086194,  0.24463256, -0.173559  ],
    [ 0.88430543,  0.56434887, -0.38659226],
    [ 0.42874197, -2.25138582,  1.23796113],
    [ 0.65319323, -1.83110915,  1.00374764],
    [ 0.29719174,  2.44516787, -1.25888844],
    [ 0.081416  ,  2.69238367, -1.3710235 ]
])

tvec_array = np.array([
    [ 0.0206323 , -0.00383275,  0.30145854],
    [ 0.00458551,  0.00155877,  0.30454052],
    [-0.03331526, -0.0206328 ,  0.30816127],
    [-0.00935626, -0.00771348,  0.30490998],
    [-0.03663627, -0.01501833,  0.30628539],
    [-0.03020911, -0.01789318,  0.3063375 ],
    [-0.00614012, -0.0159782 ,  0.30126883],
    [ 0.02093393, -0.00713036,  0.3027663 ],
    [-0.01678872, -0.01117937,  0.3020146 ],
    [-0.01164613, -0.01523511,  0.30240287]
])

xyz_array = np.array([
    [-0.19454681, -0.16880477, -0.15803204],
    [-0.28055308, -0.05991437, -0.10231630],
    [ 0.29931063,  0.0698563 ,  0.04508026],
    [ 0.25683752,  0.14840262,  0.07161201],
    [ 0.12087679, -0.21963156, -0.18035877],
    [ 0.20848446, -0.16123682, -0.16003855],
    [-0.18228401,  0.21009075,  0.11700567],
    [-0.26308367,  0.12964978,  0.07832252],
    [ 0.11131714,  0.23497005,  0.15497493],
    [ 0.02774024,  0.25270238,  0.16488548]
])
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def zoom_factory(ax, base_scale=2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0]) * .5
        cur_yrange = (cur_ylim[1] - cur_ylim[0]) * .5
        xdata = event.xdata  # get event x location
        ydata = event.ydata  # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print(event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange * scale_factor,
                     xdata + cur_xrange * scale_factor])
        ax.set_ylim([ydata - cur_yrange * scale_factor,
                     ydata + cur_yrange * scale_factor])
        plt.draw()  # force re-draw

    fig = ax.get_figure()  # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event', zoom_fun)
    # return the function
    return zoom_fun


plt.style.use('default')
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# 원점
ax.scatter(0, 0, 0, marker='o', color='k', s=20)
ax.set_xlim([-0.7, 0.7])
ax.set_xlabel('X')
ax.set_ylim([-0.7, 0.7])
ax.set_ylabel('Y')
ax.set_zlim([-0.7, 0.7])
ax.set_zlabel('Z')
scale = 1.5
f = zoom_factory(ax, base_scale=scale)

for i, blob in enumerate(object_points):
    ax.scatter(blob[0], blob[1], blob[2], marker='o', s=5, color='blue', alpha=0.5)
    label = '%s' % i
    ax.text(blob[0], blob[1], blob[2], label, size=5)

# Calculate and plot camera positions
idx = 0
for rvec, tvec in zip(rvec_array, tvec_array):
    R, _ = cv2.Rodrigues(rvec)
    cam_pos = -R.T @ tvec
    X, Y, Z = cam_pos.flatten()
    # 카메라 방향 계산
    camera_direction = R.T @ np.array([[0, 0, -1]]).T
    camera_direction = camera_direction.ravel()
    
    # 카메라 optical axis 방향 벡터 계산
    optical_axis = R.T @ np.array([0, 0, -1])

    # 카메라 위치에서 optical axis까지의 방향 벡터 계산
    direction_vector = -optical_axis
    
        # calculate roll, pitch, yaw
    roll = math.atan2(R[2][1], R[2][2])
    pitch = math.atan2(-R[2][0], math.sqrt(R[2][1]**2 + R[2][2]**2))
    yaw = math.atan2(R[1][0], R[0][0])

    # convert to degrees
    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)

    # print roll, pitch, yaw
    print(f"Camera {idx}: Roll={roll:.2f}, Pitch={pitch:.2f}, Yaw={yaw:.2f}")
        
    ax.quiver(X, Y, Z, direction_vector[0], direction_vector[1], direction_vector[2],
    color="blue", length=0.1, normalize=True)
       
    ax.scatter(X, Y, Z, marker='o')
    ax.text(X, Y, Z, f"{idx},({X:.2f}, {Y:.2f}, {Z:.2f})", fontsize=9)
    idx += 1
        
    # plot the plane perpendicular to camera direction vector
    v = direction_vector / np.linalg.norm(direction_vector)
    p0 = cam_pos
    d = 0.0  # distance from camera position to the plane
    u = np.array([1, 0, 0])
    if np.abs(np.dot(u, v)) == 1:
        u = np.array([0, 1, 0])
    w = np.cross(u, v)
    u = np.cross(v, w)
    u /= np.linalg.norm(u)
    w /= np.linalg.norm(w)

    # define vertices of the rectangle
    l = 0.1
    p1 = p0 + (d * v) + (l / 2) * (u + w)
    p2 = p0 + (d * v) + (l / 2) * (-u + w)
    p3 = p0 + (d * v) + (l / 2) * (-u - w)
    p4 = p0 + (d * v) + (l / 2) * (u - w)
    vertices = np.array([p1, p2, p3, p4])

    # plot the rectangle
    rect = Poly3DCollection([vertices], alpha=0.3, facecolor='gray', edgecolor='none')
    ax.add_collection3d(rect)
    ax.set_box_aspect([1280/960, 1, 1])

    # plot roll, pitch, yaw annotations
    # ax.text(X + 0.2, Y + 0.2, Z, f"Roll: {roll:.2f}", fontsize=9, color='red')
    # ax.text(X + 0.2, Y, Z + 0.2, f"Pitch: {pitch:.2f}", fontsize=9, color='red')
    # ax.text(X, Y + 0.2, Z + 0.2, f"Yaw: {yaw:.2f}", fontsize=9, color='red')




# import cv2
# import numpy as np

# distorted_2d = np.array([
#     [703.6601867983283, 496.0493244163074],
#     [671.4857026341083, 455.1010996108903],
#     [604.0156704653562, 504.3090024931503],
#     [548.6472817803842, 453.16743907462677]
# ])

# camera_matrix = np.array([
#     [716.229, 0, 661.222],
#     [0, 716.229, 493.271],
#     [0, 0, 1]
# ])

# dist_coeffs = np.array([0.075759, -0.026343, 0.006198, -0.000643])

# # 사용할 3D object_points 선택
# object_points_selected = object_points[[3, 4, 5, 6], :]

# retval, rvec, tvec, inliers = cv2.solvePnPRansac(object_points_selected, distorted_2d, camera_matrix, dist_coeffs)

# print('rvec', rvec, 'tvec', tvec)

plt.show()