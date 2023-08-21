import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Advanced_Calibration import *

#-------------------------------------------------------------------------------
# Generate points on circle
# P(t) = r*cos(t)*u + r*sin(t)*(n x u) + C
#-------------------------------------------------------------------------------
def generate_circle_by_vectors(t, C, r, n, u):
    n = n / np.linalg.norm(n)
    u = u / np.linalg.norm(u)
    
    P_circle = (r * np.cos(t)[:,np.newaxis] * u) + (r * np.sin(t)[:,np.newaxis] * np.cross(n,u)) + C
    return P_circle

def generate_circle_by_angles(t, C, r, theta, phi):
    # Orthonormal vectors n, u, <n,u>=0
    n = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    u = np.array([-np.sin(phi), np.cos(phi), 0])
    
    # P(t) = r*cos(t)*u + r*sin(t)*(n x u) + C
    P_circle = r*np.cos(t)[:,np.newaxis]*u + r*np.sin(t)[:,np.newaxis]*np.cross(n,u) + C
    return P_circle

#-------------------------------------------------------------------------------
# Generating circle
#-------------------------------------------------------------------------------
# r = 2.5               # Radius
# C = np.array([3,3,4])    # Center
# theta = 45/180*np.pi     # Azimuth
# phi   = -30/180*np.pi     # Zenith

# t = np.linspace(0, 2*np.pi , 100)
# P_gen = generate_circle_by_angles(t, C, r, theta, phi)

#-------------------------------------------------------------------------------
# Cluster of points
#-------------------------------------------------------------------------------
# t = np.linspace(-np.pi, -0.25*np.pi, 100)
# n = len(t)
# P = generate_circle_by_angles(t, C, r, theta, phi)

# # Add some random noise to the points
# P += np.random.normal(size=P.shape) * 0.1

P = np.vstack(pickle_data(READ, 'BLENDER.pickle', None)['opositions'])
# bpositions = np.vstack(pickle_data(READ, 'BLENDER.pickle', None)['bpositions'])

#-------------------------------------------------------------------------------
# Plot
#-------------------------------------------------------------------------------
f, ax = plt.subplots(1, 3, figsize=(15,5))
alpha_pts = 0.5
i = 0
# ax[i].plot(P_gen[:,0], P_gen[:,1], 'y-', lw=3, label='Generating circle')
ax[i].scatter(P[:,0], P[:,1], alpha=alpha_pts, label='Cluster points P')
ax[i].set_title('View X-Y')
ax[i].set_xlabel('x'); ax[i].set_ylabel('y');
ax[i].set_aspect('equal', 'datalim'); ax[i].margins(.1, .1)
ax[i].grid()
i = 1
# ax[i].plot(P_gen[:,0], P_gen[:,2], 'y-', lw=3, label='Generating circle')
ax[i].scatter(P[:,0], P[:,2], alpha=alpha_pts, label='Cluster points P')
ax[i].set_title('View X-Z')
ax[i].set_xlabel('x'); ax[i].set_ylabel('z'); 
ax[i].set_aspect('equal', 'datalim'); ax[i].margins(.1, .1)
ax[i].grid()
i = 2
# ax[i].plot(P_gen[:,1], P_gen[:,2], 'y-', lw=3, label='Generating circle')
ax[i].scatter(P[:,1], P[:,2], alpha=alpha_pts, label='Cluster points P')
ax[i].set_title('View Y-Z')
ax[i].set_xlabel('y'); ax[i].set_ylabel('z'); 
ax[i].set_aspect('equal', 'datalim'); ax[i].margins(.1, .1)
ax[i].legend()
ax[i].grid()

#-------------------------------------------------------------------------------
# FIT CIRCLE 2D
# - Find center [xc, yc] and radius r of circle fitting to set of 2D points
# - Optionally specify weights for points
#
# - Implicit circle function:
#   (x-xc)^2 + (y-yc)^2 = r^2
#   (2*xc)*x + (2*yc)*y + (r^2-xc^2-yc^2) = x^2+y^2
#   c[0]*x + c[1]*y + c[2] = x^2+y^2
#
# - Solution by method of least squares:
#   A*c = b, c' = argmin(||A*c - b||^2)
#   A = [x y 1], b = [x^2+y^2]
#-------------------------------------------------------------------------------
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


#-------------------------------------------------------------------------------
# RODRIGUES ROTATION
# - Rotate given points based on a starting and ending vector
# - Axis k and angle of rotation theta given by vectors n0,n1
#   P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
#-------------------------------------------------------------------------------
def rodrigues_rot(P, n0, n1):
    
    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[np.newaxis,:]
    
    # Get vector of rotation k and angle theta
    n0 = n0/np.linalg.norm(n0)
    n1 = n1/np.linalg.norm(n1)
    k = np.cross(n0,n1)
    k = k/np.linalg.norm(k)
    theta = np.arccos(np.dot(n0,n1))
    
    # Compute rotated points
    P_rot = np.zeros((len(P),3))
    for i in range(len(P)):
        P_rot[i] = P[i]*np.cos(theta) + np.cross(k,P[i])*np.sin(theta) + k*np.dot(k,P[i])*(1-np.cos(theta))

    return P_rot


#-------------------------------------------------------------------------------
# ANGLE BETWEEN
# - Get angle between vectors u,v with sign based on plane with unit normal n
#-------------------------------------------------------------------------------
def angle_between(u, v, n=None):
    if n is None:
        return np.arctan2(np.linalg.norm(np.cross(u,v)), np.dot(u,v))
    else:
        return np.arctan2(np.dot(n,np.cross(u,v)), np.dot(u,v))

    
#-------------------------------------------------------------------------------
# - Make axes of 3D plot to have equal scales
# - This is a workaround to Matplotlib's set_aspect('equal') and axis('equal')
#   which were not working for 3D
#-------------------------------------------------------------------------------
def set_axes_equal_3d(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = abs(limits[:,0] - limits[:,1])
    centers = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)
    ax.set_xlim3d([centers[0]-radius, centers[0]+radius])
    ax.set_ylim3d([centers[1]-radius, centers[1]+radius])
    ax.set_zlim3d([centers[2]-radius, centers[2]+radius])

#-------------------------------------------------------------------------------
# Init figures
#-------------------------------------------------------------------------------
fig = plt.figure(figsize=(15,11))
alpha_pts = 0.5
figshape = (2,3)
ax = [None]*4
ax[0] = plt.subplot2grid(figshape, loc=(0,0), colspan=2)
ax[1] = plt.subplot2grid(figshape, loc=(1,0))
ax[2] = plt.subplot2grid(figshape, loc=(1,1))
ax[3] = plt.subplot2grid(figshape, loc=(1,2))
i = 0
ax[i].set_title('Fitting circle in 2D coords projected onto fitting plane')
ax[i].set_xlabel('x'); ax[i].set_ylabel('y');
ax[i].set_aspect('equal', 'datalim'); ax[i].margins(.1, .1); ax[i].grid()
i = 1
# ax[i].plot(P_gen[:,0], P_gen[:,1], 'y-', lw=3, label='Generating circle')
ax[i].scatter(P[:,0], P[:,1], alpha=alpha_pts, label='Cluster points P')
ax[i].set_title('View X-Y')
ax[i].set_xlabel('x'); ax[i].set_ylabel('y');
ax[i].set_aspect('equal', 'datalim'); ax[i].margins(.1, .1); ax[i].grid()
i = 2
# ax[i].plot(P_gen[:,0], P_gen[:,2], 'y-', lw=3, label='Generating circle')
ax[i].scatter(P[:,0], P[:,2], alpha=alpha_pts, label='Cluster points P')
ax[i].set_title('View X-Z')
ax[i].set_xlabel('x'); ax[i].set_ylabel('z'); 
ax[i].set_aspect('equal', 'datalim'); ax[i].margins(.1, .1); ax[i].grid()
i = 3
# ax[i].plot(P_gen[:,1], P_gen[:,2], 'y-', lw=3, label='Generating circle')
ax[i].scatter(P[:,1], P[:,2], alpha=alpha_pts, label='Cluster points P')
ax[i].set_title('View Y-Z')
ax[i].set_xlabel('y'); ax[i].set_ylabel('z'); 
ax[i].set_aspect('equal', 'datalim'); ax[i].margins(.1, .1); ax[i].grid()

#-------------------------------------------------------------------------------
# (1) Fitting plane by SVD for the mean-centered data
# Eq. of plane is <p,n> + d = 0, where p is a point on plane and n is normal vector
#-------------------------------------------------------------------------------
P_mean = P.mean(axis=0)
P_centered = P - P_mean
U,s,V = np.linalg.svd(P_centered)

# Normal vector of fitting plane is given by 3rd column in V
# Note linalg.svd returns V^T, so we need to select 3rd row from V^T
normal = V[2,:]
d = -np.dot(P_mean, normal)  # d = -<p,n>

#-------------------------------------------------------------------------------
# (2) Project points to coords X-Y in 2D plane
#-------------------------------------------------------------------------------
P_xy = rodrigues_rot(P_centered, normal, [0,0,1])

ax[0].scatter(P_xy[:,0], P_xy[:,1], alpha=alpha_pts, label='Projected points')

#-------------------------------------------------------------------------------
# (3) Fit circle in new 2D coords
#-------------------------------------------------------------------------------

print('P_xy\n', P_xy)
# print(f"P_xy[:,0] {P_xy[:,0]}   P_xy[:,1] {P_xy[:,1]}")

xc, yc, r = fit_circle_2d(P_xy[:,0], P_xy[:,1])
print('xc ', xc, 'yc ', yc, 'r ', r)
#--- Generate circle points in 2D
t = np.linspace(0, 2*np.pi, 100)
xx = xc + r*np.cos(t)
yy = yc + r*np.sin(t)

print('xx', xx)
print('yy', yy)

ax[0].plot(xx, yy, 'k--', lw=2, label='Fitting circle')
ax[0].plot(xc, yc, 'k+', ms=10)
ax[0].legend()

#-------------------------------------------------------------------------------
# (4) Transform circle center back to 3D coords
#-------------------------------------------------------------------------------
C = rodrigues_rot(np.array([xc,yc,0]), [0,0,1], normal) + P_mean
C = C.flatten()
print('C:', C)
#--- Generate points for fitting circle
t = np.linspace(0, 2*np.pi, 100)
u = P[0] - C
# 설정할 점의 수
num_points = 120
# 원하는 만큼의 각도 값을 생성합니다. np.linspace는 주어진 범위 내에서 균등하게 분포된 값을 생성합니다.
t = np.linspace(0, 2*np.pi, num_points)
print('normal ', normal)
P_fitcircle = generate_circle_by_vectors(t, C, r, normal, u)


print('P_fitcircle\n', P_fitcircle)

ax[1].plot(P_fitcircle[:,0], P_fitcircle[:,1], 'k--', lw=2, label='Fitting circle')
ax[2].plot(P_fitcircle[:,0], P_fitcircle[:,2], 'k--', lw=2, label='Fitting circle')
ax[3].plot(P_fitcircle[:,1], P_fitcircle[:,2], 'k--', lw=2, label='Fitting circle')
ax[3].legend()

#--- Generate points for fitting arc
u = P[0] - C
v = P[-1] - C
theta = angle_between(u, v, normal)

t = np.linspace(0, theta, 100)
P_fitarc = generate_circle_by_vectors(t, C, r, normal, u)

ax[1].plot(P_fitarc[:,0], P_fitarc[:,1], 'k-', lw=3, label='Fitting arc')
ax[2].plot(P_fitarc[:,0], P_fitarc[:,2], 'k-', lw=3, label='Fitting arc')
ax[3].plot(P_fitarc[:,1], P_fitarc[:,2], 'k-', lw=3, label='Fitting arc')
ax[1].plot(C[0], C[1], 'k+', ms=10)
ax[2].plot(C[0], C[2], 'k+', ms=10)
ax[3].plot(C[1], C[2], 'k+', ms=10)
ax[3].legend()

Fitting_plane = normal
Fitting_circle = np.array_str(C, precision=4)
Fitting_circle_r = r
print('Fitting plane: n = %s' % np.array_str(normal, precision=4))
print('Fitting circle: center = %s, r = %.4g' % (Fitting_circle, Fitting_circle_r))
print('Fitting arc: u = %s, θ = %.4g' % (np.array_str(u, precision=4), math.degrees(theta*180/np.pi)))

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(1,1,1,projection='3d')
# ax.plot(*P_gen.T, color='y', lw=3, label='Generating circle')
ax.plot(*P.T, ls='', marker='o', alpha=0.5, label='Cluster points P')

#--- Plot fitting plane
xx, yy = np.meshgrid(np.linspace(0,6,11), np.linspace(0,6,11))
zz = (-normal[0]*xx - normal[1]*yy - d) / normal[2]
ax.plot_surface(xx, yy, zz, rstride=2, cstride=2, color='y' ,alpha=0.2, shade=False)

#--- Plot fitting circle
ax.plot(*P_fitcircle.T, color='k', ls='--', lw=2, label='Fitting circle')
ax.plot(*P_fitarc.T, color='k', ls='-', lw=3, label='Fitting arc')
# Plotting the vector as an arrow
ax.quiver(*C, *normal, color='b', arrow_length_ratio=0.1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

ax.set_aspect('auto', 'datalim')
set_axes_equal_3d(ax)

scale = 1.5
f = zoom_factory(ax, base_scale=scale)


data = OrderedDict()
data['P_fitcircle'] = P_fitcircle
data['CENTER'] = C
data['R'] = r
data['Fitting_plane'] = Fitting_plane
pickle_data(WRITE, 'FITTING_CIRCLE.pickle', data)

plt.show()