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

import numpy as np
from scipy.spatial.distance import pdist

def fit_circle_2d_fixed_center(x, y, center, w=[]):
    # Use the provided center
    xc, yc = center

    # Calculate radius based on the provided center
    r = np.mean(np.sqrt((x - xc)**2 + (y - yc)**2))

    # If provided weights, adjust the radius accordingly
    if len(w) == len(x):
        r = np.sum(w * np.sqrt((x - xc)**2 + (y - yc)**2)) / np.sum(w)

    return xc, yc, r


from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt

SOLUTION = 3

#-------------------------------------------------------------------------------
# Init figures
#-------------------------------------------------------------------------------
#--------

# TEST 1
P = np.array([[641, 612],
                   [693, 605],
                   [600, 602],
                   [635, 587],
                   [688, 584],
                   [598, 579],
                   [719, 573],
                   [572, 561]])

#TEST 2
# P = np.array([
# [590, 603],
# [688, 600],
# [569, 592],
# [567, 569],
# [710, 565],
# [550, 552]])

# P = np.array([
# [639, 615],
# [665, 591],
# [628, 590],
# [707, 581],
# [584, 575],
# [733, 566],
# [565, 557]])


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

# P = np.array([
# [680, 609],
# [583, 602],
# [704, 599],
# [565, 589],
# [705, 577],
# [564, 567],
# [729, 558]
# ])

P = np.array([
[671, 614],
[628, 612],
[717, 599],
[591, 598],
[667, 593],
[627, 591],
[715, 581],
[590, 577],
[738,566],
[572,557]
])
# P = np.array([
# [637, 619],
# [668, 617],
# [574, 604],
# [634, 597],
# [597, 593],
# [565, 578]
# ])

# P = np.array([
# [675, 612],
# [632, 611],
# [712, 599],
# [592, 597],
# [677, 590],
# [632, 590],
# [573, 584],
# [712, 580],
# [595, 578],
# [740, 556]
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
    Base_center = np.array([bxc, byc])

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



    distances = np.linalg.norm(P - Inner_center, axis=1)    
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

elif SOLUTION == 3:
    t = np.linspace(0, 2*np.pi, 100)
    ax[0].scatter(P[:,0], P[:,1], alpha=0.5, label='Projected points', color='gray')
    # MAKE Base CIRCLE to find General Center Point
    if len(P) > 5:
        bxc, byc, br = fit_circle_2d(P[:,0], P[:,1])
    else:
        bxc, byc, br = fit_circle_2d_fixed_center(P[:,0], P[:,1], center=(640,480))
    print('bxc ', bxc, 'byc ', byc, 'br ', br)
    DELTA = 0
    if len(P) > 5:
        DELTA = -br*1.5
    Base_center = np.array([bxc, byc + DELTA])

    bxx = bxc + br*np.cos(t)
    byy = byc + br*np.sin(t)
    ax[0].plot(bxx, byy, 'k--', lw=2, label='Fitting circle', color='gray')
    ax[0].plot(bxc, byc, 'k+', ms=10, color='gray')
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
    inside_points = [] # List to hold the points that are inside the circle for each label

    # For each cluster
    for i in range(n_clusters):
        # Get the points in this cluster
        cluster_points = P[labels == i]
        # Calculate the distance from the center of the circle to each point in this cluster
        distances_to_center = np.sqrt((cluster_points[:, 0] - bxc)**2 + (cluster_points[:, 1] - byc)**2)
        # Find the points where the distance is less than or equal to the radius
        inside = cluster_points[distances_to_center <= br]        
        # Add these points to the list
        inside_points.append(inside)        
        # Plot the points
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1])
    # Print the points that are inside the circle for each label
    for i in range(n_clusters):
        print(f"Points inside the circle for label {i}: {inside_points[i]}")    
    # Convert the list of arrays into a single numpy array
    inside_points_arr = np.concatenate(inside_points, axis=0)
    
    xc, yc, r = fit_circle_2d(inside_points_arr[:,0], inside_points_arr[:,1])
    print('xc ', xc, 'yc ', yc, 'r ', r)
    Inner_center = np.array([xc, yc])

    xx = xc + r*np.cos(t)
    yy = yc + r*np.sin(t)
    ax[0].scatter(inside_points_arr[:,0], inside_points_arr[:,1], alpha=0.5, label='Projected points', color='red')
    ax[0].plot(xx, yy, 'k--', lw=2, label='Fitting circle', color='red')
    ax[0].plot(xc, yc, 'k+', ms=10, color='red')
    ax[0].legend()
    
    
    distances = np.linalg.norm(P - Inner_center, axis=1)    
    distances = distances.reshape(-1, 1)
    print('distance ', distances)

    n_clusters = 2  # Change this value according to your requirement

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
    labels = clustering.fit_predict(distances)
    print('labels ', labels)
    if len(P) <= 5:
        for i in range(n_clusters):
            plt.scatter(P[labels == i, 0], P[labels == i, 1])
    plt.show()

