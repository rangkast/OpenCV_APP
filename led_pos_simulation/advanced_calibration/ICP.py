import numpy as np
import cv2
import open3d as o3d


def add_noise(points, sigma=0.1):
    # Add Gaussian noise to each point
    noise = np.random.normal(0, sigma, points.shape)
    points += noise
    return points


def project_2d(points, P):
    # Apply transformation and project points to 2D
    points = (P @ np.append(points, np.ones((points.shape[0], 1)), axis=1).T).T
    return points[:, :2] / points[:, 2, np.newaxis]


def triangulate_points(points2D_1, points2D_2, P1, P2):
    # Convert points to homogeneous coordinates
    points2D_1_hom = np.vstack((points2D_1.T, np.ones(points2D_1.shape[0])))
    points2D_2_hom = np.vstack((points2D_2.T, np.ones(points2D_2.shape[0])))

    # Triangulate points
    points3D_hom = cv2.triangulatePoints(P1, P2, points2D_1_hom[:2], points2D_2_hom[:2])

    # Convert back to 3D
    points3D = points3D_hom[:3] / points3D_hom[3]

    return points3D.T


# Initial 3D points
points3D = np.array([[-5.0, 1.0, 1.0], [-5.0, 1.0, -1.0], [-5.0, -1.0, 1.0], [-5.0, -1.0, -1.0]])

# Add noise
points3D_noisy = add_noise(points3D.copy())

# Define camera matrices P1 and P2 for two views
P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
P2 = np.hstack([np.eye(3), np.array([5, 0, 0]).reshape(3, 1)])

# Project to 2D
points2D_1 = project_2d(points3D_noisy, P1)
points2D_2 = project_2d(points3D_noisy, P2)

# Recover 3D
points3D_recovered = triangulate_points(points2D_1, points2D_2, P1, P2)

# Calculate Euclidean distance using numpy
distance = np.sqrt(np.sum((points3D_noisy - points3D_recovered) ** 2, axis=1))
print(f"Euclidean distance: {distance}")
