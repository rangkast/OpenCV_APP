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

from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt

points = np.array([[641, 612],
                   [693, 605],
                   [600, 602],
                   [635, 587],
                   [688, 584],
                   [598, 579],
                   [719, 573],
                   [572, 561]])

center = np.array([642, 482])

distances = np.linalg.norm(points - center, axis=1)
distances = distances.reshape(-1, 1)

n_clusters = 2  # Change this value according to your requirement

clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
labels = clustering.fit_predict(distances)

for i in range(n_clusters):
    plt.scatter(points[labels == i, 0], points[labels == i, 1])

plt.scatter(center[0], center[1], c='black')  # plot the center
plt.gca().invert_yaxis()
plt.show()
