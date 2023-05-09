import numpy as np
import matplotlib.pyplot as plt

A = np.array([
    [420.87628866, 308.80412371],
    [465.88888889, 355.85185185],
    [528.71770335, 335.22727273],
    [661.53773963, 303.18614789]
])

B = np.array([
    [421.95564686, 309.26589251],
    [466.5565915,  356.48493308],
    [529.21278897, 335.63039194],
    [663.13104057, 303.08079013]
])

plt.scatter(A[:, 0], A[:, 1], c='blue', label='A')
plt.scatter(B[:, 0], B[:, 1], c='red', label='B')

# 각 점에 레이블 표시
for i, (x, y) in enumerate(A):
    plt.text(x, y, f'A{i}', fontsize=12, ha='right', va='bottom')

for i, (x, y) in enumerate(B):
    plt.text(x, y, f'B{i}', fontsize=12, ha='right', va='bottom')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot of Coordinate Groups A and B')
plt.legend()
plt.grid()
plt.show()
