import cv2
import numpy as np
from numpy.linalg import inv
from numpy import shape, tile, matmul, zeros, transpose

# SIMULATE SOME POINTS
# Rotation Matrix
R = np.eye(3)
# Translation vector
t = np.array([[1],[0],[0]])
# Camera calibration matrix
K = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
# World frame coordinates of the points
Xw = np.array([[0,0,0,0], [1,1,1,0], [2,2,2,0], [3,3,3,0], [4,4,4,0], [5,5,5,0]]).T
Xw = Xw.astype('float64')
# [R|t]
Rt = np.concatenate((R,t), axis=1)
# Distortion coefficients
dist_coeffs = np.array([1, 1, 1, 1, 1])

# GET THE JACOBIAN TO SOLVE THE LEAST SQUARES EQUATION FOR t
# Transpose Xw to fit projectPoints()
Xw = Xw[:3,:].T
# Compute the Jacobian
_, H = cv2.projectPoints(Xw, R, t, K, dist_coeffs)