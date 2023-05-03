import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 타원체의 파라미터
r1 = 5  # +Z 축 방향의 반경
r2 = 3  # -Z 축 방향의 반경

# 타원체의 격자 점 생성
u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(-np.pi/2, np.pi/2, 100)
x = r1 * np.outer(np.cos(u), np.cos(v))
y = r1 * np.outer(np.sin(u), np.cos(v))
z1 = r2 * np.outer(np.ones(np.size(u)), np.sin(v))
z2 = r2 * np.outer(np.ones(np.size(u)), -np.sin(v))

# 배열의 모양을 맞춰줍니다.
x, y, z1 = np.broadcast_arrays(x, y, z1)
x, y, z2 = np.broadcast_arrays(x, y, z2)

# z1, z2 배열의 차원을 (n, m, 1)로 바꿔줍니다.
z1 = z1[:, :, np.newaxis]
z2 = z2[:, :, np.newaxis]

# z1, z2 배열을 하나의 배열로 합쳐줍니다.
z = np.concatenate((z1, z2), axis=2)

# 3D 그래프 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# z 배열의 첫 번째 차원 제거하여 2차원 배열 생성
z_2d = np.squeeze(z)

# 타원체 시각화
ax.plot_surface(x, y, z_2d, alpha=0.5)

# 그래프 축 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 그래프 표시
plt.show()
