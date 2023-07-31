# from sklearn.cluster import DBSCAN
# import matplotlib.pyplot as plt
# import numpy as np

# # Define the points

# points = np.array([
#     [641, 612],
#     [693, 605],
#     [600, 602],
#     [635, 587],
#     [688, 584],
#     [598, 579],
#     [719, 573],
#     [572, 561]
# ])
# points[:, 1] = 960 - points[:, 1] # adjust y-coordinate to image coordinate system

# # Fit the DBSCAN clusterer to the data
# clusterer = DBSCAN(eps=50, min_samples=2) # these parameters may need to be adjusted depending on your data
# labels = clusterer.fit_predict(points)

# # Plot the points color coded by cluster
# for i in range(max(labels)+1):
#     cluster_points = points[labels == i]
#     print(cluster_points)
#     plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')

# # Mark the noise points (those not in any cluster) in a different color and shape
# noise_points = points[labels == -1]
# plt.scatter(noise_points[:, 0], noise_points[:, 1], c='black', marker='x', label='Noise')

# # invert y-axis to match the image coordinate system
# plt.gca().invert_yaxis()

# plt.legend()
# plt.show()

def fit_circle_2d(x, y, w=[]):
    
    A = np.array([x, y, np.ones(len(x))]).T
    b = x**2 + y**2
    
    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W,A)
        b = np.dot(W,b)
    
    # Solve by method of least squares
    c = np.linalg.lstsq(A,b,rcond=None)[0]
    
    # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r

from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt

SOLUTION = 2

#-------------------------------------------------------------------------------
# Init figures
#-------------------------------------------------------------------------------
#--------

# TEST 1
# P = np.array([[641, 612],
#                    [693, 605],
#                    [600, 602],
#                    [635, 587],
#                    [688, 584],
#                    [598, 579],
#                    [719, 573],
#                    [572, 561]])

#TEST 2
# P = np.array([
# [590, 603],
# [688, 600],
# [569, 592],
# [567, 569],
# [710, 565],
# [550, 552]])

P = np.array([
[639, 615],
[665, 591],
[628, 590],
[707, 581],
[584, 575],
[733, 566],
[565, 557]])


# P = np.array([
# [631, 606],
# [678, 604],
# [602, 601],
# [635, 583],
# [678, 580],
# [715, 566]
# ])

# 이게 문제임!
# 안됨
# P = np.array([
# [639, 608],
# [669, 604],
# [672, 580],
# [709, 568]
# ])

# P = np.array([
# [613, 610],
# [587, 603],
# [584, 580],
# [558, 566],
# [722, 555]
# ])

# P = np.array([
# [639, 621],
# [600, 615],
# [705, 602],
# [627, 596],
# [669, 593],
# [724, 588],
# [585, 587],
# [701, 581],
# [557, 566]
# ])
fig = plt.figure(figsize=(15,11))
alpha_pts = 0.5
figshape = (2,3)
ax = [None]*4
ax[0] = plt.subplot2grid(figshape, loc=(0,0), colspan=2)
ax[1] = plt.subplot2grid(figshape, loc=(1,0))

i = 0
ax[i].set_title('Fitting circle in 2D coords projected onto fitting plane')
ax[i].set_xlabel('x'); ax[i].set_ylabel('y')
ax[i].set_aspect('equal', 'datalim'); ax[i].margins(.1, .1); ax[i].grid()
ax[i].invert_yaxis()
i = 1
# ax[i].plot(P_gen[:,0], P_gen[:,1], 'y-', lw=3, label='Generating circle')
ax[i].scatter(P[:,0], P[:,1], alpha=alpha_pts, label='Cluster points P')
ax[i].set_title('View X-Y')
ax[i].set_xlabel('x'); ax[i].set_ylabel('y')
ax[i].set_aspect('equal', 'datalim'); ax[i].margins(.1, .1); ax[i].grid()
ax[i].invert_yaxis()

if SOLUTION == 1:
    xc, yc, r = fit_circle_2d(P[:,0], P[:,1])
    print('xc ', xc, 'yc ', yc, 'r ', r)
    ax[0].scatter(P[:,0], P[:,1], alpha=0.5, label='Projected points')
    #--- Generate circle points in 2D
    t = np.linspace(0, 2*np.pi, 100)
    xx = xc + r*np.cos(t)
    yy = yc + r*np.sin(t)

    print('xx', xx)
    print('yy', yy)

    ax[0].plot(xx, yy, 'k--', lw=2, label='Fitting circle')
    ax[0].plot(xc, yc, 'k+', ms=10)
    ax[0].legend()
    plt.show()

elif SOLUTION == 2:
    t = np.linspace(0, 2*np.pi, 100)
    ax[0].scatter(P[:,0], P[:,1], alpha=0.5, label='Projected points', color='gray')
    # MAKE Base CIRCLE to find General Center Point
    bxc, byc, br = fit_circle_2d(P[:,0], P[:,1])
    print('bxc ', bxc, 'byc ', byc, 'br ', br)
    Base_center = np.array([bxc, byc - 100])

    bxx = bxc + br*np.cos(t)
    byy = byc + br*np.sin(t)
    ax[0].plot(bxx, byy, 'k--', lw=2, label='Fitting circle', color='gray')
    ax[0].plot(bxc, byc, 'k+', ms=10, color='gray')
    ax[0].legend()

    POINTS = []
    for points in P:
        POINTS.append([points[0], points[1], np.linalg.norm(points - Base_center, axis=0)])

    POINTS_SORTED = np.array(sorted(POINTS, key=lambda x:x[2])) ## 또는 l.sort(key=lambda x:x[1])
    print('POINTS_SORTED\n', POINTS_SORTED)

    # MAKE Inner Circle made by 3 closest Points
    POINTS_SORTED = POINTS_SORTED[:3]
    print('POINTS_SORTED\n', POINTS_SORTED)
    ax[0].scatter(POINTS_SORTED[:,0], POINTS_SORTED[:,1], alpha=0.5, label='Projected points', color='red')

    xc, yc, r = fit_circle_2d(POINTS_SORTED[:,0], POINTS_SORTED[:,1])
    print('xc ', xc, 'yc ', yc, 'r ', r)
    Inner_center = np.array([xc, yc])

    xx = xc + r*np.cos(t)
    yy = yc + r*np.sin(t)

    ax[0].plot(xx, yy, 'k--', lw=2, label='Fitting circle', color='red')
    ax[0].plot(xc, yc, 'k+', ms=10, color='red')
    ax[0].legend()



    distances = np.linalg.norm(P - Base_center, axis=1)    
    distances = distances.reshape(-1, 1)
    print('distance ', distances)

    n_clusters = 2  # Change this value according to your requirement

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    labels = clustering.fit_predict(distances)
    print('labels ', labels)
    for i in range(n_clusters):
        plt.scatter(P[labels == i, 0], P[labels == i, 1])

    plt.scatter(Base_center[0], Base_center[1], c='gray')
    plt.scatter(Inner_center[0], Inner_center[1], c='red')
    plt.show()


