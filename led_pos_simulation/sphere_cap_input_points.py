import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_spherical_caps_with_points(R, h, num_points, num_surface_points):
    # 데이터 간격 설정
    num_theta_points = int(np.sqrt(num_surface_points))
    num_phi_points = num_surface_points // num_theta_points

    # 구 캡 위의 점들을 계산
    theta = np.linspace(0, 2 * np.pi, num_theta_points)
    phi_upper = np.linspace(np.arccos(1 - h / R), np.pi/2, num_phi_points)
    phi_lower = np.linspace(np.pi/2, np.arccos(1 - h / R), num_phi_points)
    theta_upper, phi_upper = np.meshgrid(theta, phi_upper)
    theta_lower, phi_lower = np.meshgrid(theta, phi_lower)

    x_upper = R * np.sin(phi_upper) * np.cos(theta_upper)
    y_upper = R * np.sin(phi_upper) * np.sin(theta_upper)
    z_upper = R * np.cos(phi_upper)

    x_lower = R * np.sin(phi_lower) * np.cos(theta_lower)
    y_lower = R * np.sin(phi_lower) * np.sin(theta_lower)
    z_lower = R * np.cos(phi_lower)

    # 3D 플롯 생성
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 구 캡을 3D 플롯에 추가
    ax.plot_surface(x_upper, y_upper, z_upper, color='lightgray', alpha=0.5)
    ax.plot_surface(x_lower, y_lower, z_lower, color='lightgray', alpha=0.5)

    # 표면 위의 점들을 계산
    theta_points = np.linspace(0, 2 * np.pi, num_points)
    phi_points_upper = np.arccos(1 - h / R) * np.ones_like(theta_points)
    phi_points_lower = (np.pi - np.arccos(1 - h / R)) * np.ones_like(theta_points)

    x_points_upper = R * np.sin(phi_points_upper) * np.cos(theta_points)
    y_points_upper = R * np.sin(phi_points_upper) * np.sin(theta_points)
    z_points_upper = R * np.cos(phi_points_upper)

    x_points_lower = R * np.sin(phi_points_lower) * np.cos(theta_points)
    y_points_lower = R * np.sin(phi_points_lower) * np.sin(theta_points)
    z_points_lower = R * np.cos(phi_points_lower)

    # 점들을 3D 플롯에 추가
    ax.scatter(x_points_upper, y_points_upper, z_points_upper, c='b', marker='o')
    ax.scatter(x_points_lower, y_points_lower, z_points_lower, c='r', marker='o')

    # 플롯 옵션 설정
    ax.set_title('Spherical Caps with Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # x, y, z
    axis_limit = R * 1.5
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-axis_limit, axis_limit)
    ax.set_zlim(-axis_limit, axis_limit)
    # 결과 표시
    plt.show()


R = 11.5
h = 3
num_points = 24
num_surface_points = 60
plot_spherical_caps_with_points(R, h, num_points, num_surface_points)