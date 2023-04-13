import numpy as np
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # 카메라 위치와 방향 설정
        self.cam.setPos(0, -10, 5)
        self.look_at = Point3(0, 0, 0)
        self.cam.lookAt(self.look_at)


        # 그리드 생성
        self.create_grid()

        self.points = self.generate_points(radius=1, num_points=100, lower_z=-0.5, upper_z=0.5)
        self.draw_points()

      
        self.mouse_drag_start = None
        self.azim_start = 0
        self.elev_start = 0

        self.accept("mouse1", self.on_mouse_click)
        self.accept("mouse1-up", self.on_mouse_release)

        taskMgr.add(self.update_camera_rotation, "UpdateCameraRotation")

    def on_mouse_click(self):
        self.mouse_drag_start = self.mouseWatcherNode.getMouse()
        self.azim_start, self.elev_start = self.get_camera_rotation()

    def on_mouse_release(self):
        self.mouse_drag_start = None

    def get_camera_rotation(self):
        cam_vec = self.cam.getPos() - self.look_at
        cam_vec.normalize()
        azim = np.degrees(np.arctan2(cam_vec.getY(), cam_vec.getX()))
        elev = np.degrees(np.arctan2(cam_vec.getZ(), np.sqrt(cam_vec.getX() ** 2 + cam_vec.getY() ** 2)))
        return azim, elev

    def update_camera_rotation(self, task):
        if self.mouse_drag_start is not None:
            if self.mouseWatcherNode.hasMouse():
                mouse_pos = self.mouseWatcherNode.getMouse()
                mouse_delta = self.mouse_drag_start - mouse_pos

                azim = self.azim_start - mouse_delta.getX() * 180
                elev = self.elev_start + mouse_delta.getY() * 180
                elev = max(min(elev, 89), -89)

                cam_distance = (self.cam.getPos() - self.look_at).length()
                self.cam.setPos(self.look_at + self.spherical_to_cartesian(cam_distance, azim, elev))
                self.cam.lookAt(0, 0, 0)

        return task.cont


    def spherical_to_cartesian(self, r, azim, elev):
        x = r * np.cos(np.radians(elev)) * np.cos(np.radians(azim))
        y = r * np.cos(np.radians(elev)) * np.sin(np.radians(azim))
        z = r * np.sin(np.radians(elev))
        return LVector3(x, y, z)

    
    def generate_points(self, radius, num_points, lower_z, upper_z):
        theta = np.linspace(0, 2 * np.pi, num_points)
        phi = np.linspace(0, np.pi, num_points)
        theta, phi = np.meshgrid(theta, phi)

        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        mask = (z >= lower_z) & (z <= upper_z)
        x_masked = x[mask]
        y_masked = y[mask]
        z_masked = z[mask]

        return np.vstack((x_masked, y_masked, z_masked)).T

    def draw_points(self):
        for point in self.points:
            sphere = loader.loadModel("models/misc/sphere")
            sphere.setPos(point[0], point[1], point[2])
            sphere.setScale(0.01)
            sphere.reparentTo(render)

    def create_grid(self):
        grid_size = 10
        grid_spacing = 1

        for i in range(-grid_size // 2, grid_size // 2 + 1):
            line = LineSegs()
            line.set_color(0.5, 0.5, 0.5, 1)
            line.set_thickness(1)

            line.move_to(i * grid_spacing, -grid_size // 2 * grid_spacing, 0)
            line.draw_to(i * grid_spacing, grid_size // 2 * grid_spacing, 0)

            line.move_to(-grid_size // 2 * grid_spacing, i * grid_spacing, 0)
            line.draw_to(grid_size // 2 * grid_spacing, i * grid_spacing, 0)

            node = line.create()
            self.render.attach_new_node(node)

app = MyApp()
app.run()
