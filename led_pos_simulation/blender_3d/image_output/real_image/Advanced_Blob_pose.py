
from Advanced_Function import *
import numpy as np
import subprocess
import re
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout, QGroupBox
from PyQt5.QtCore import QTimer, pyqtSignal, pyqtSlot, QThread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import datetime
from mpl_toolkits.mplot3d import Axes3D
import asyncio
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Quiver3DGraphicsView(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        super(Quiver3DGraphicsView, self).__init__(self.fig)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_box_aspect([1280/960, 1, 1])  # 고정된 3D box 비율
        self.ax.set_xlim([-1, 1])  # x축의 한계
        self.ax.set_ylim([-1, 1])  # y축의 한계
        self.ax.set_zlim([-1, 1])  # z축의 한계
        self.quiver = None

    def update_data(self, location, direction):
        if self.quiver:
            self.quiver.remove()
        
        # 길이와 화살표 머리의 크기를 조절
        X, Y, Z = location
        self.quiver = self.ax.quiver(X, Y, Z, direction[0], direction[1], direction[2], color="blue", length=0.1, normalize=True)
        self.ax.scatter(X, Y, Z, marker='o')
        
        # 방향벡터에 직교하는 평면 표현
        v = direction / np.linalg.norm(direction)
        u = np.array([1, 0, 0])
        if np.abs(np.dot(u, v)) == 1:
            u = np.array([0, 1, 0])
        w = np.cross(u, v)
        u = np.cross(v, w)
        u /= np.linalg.norm(u)
        w /= np.linalg.norm(w)
        l = 0.1
        p1 = location + (l / 2) * (u + w)
        p2 = location + (l / 2) * (-u + w)
        p3 = location + (l / 2) * (-u - w)
        p4 = location + (l / 2) * (u - w)
        vertices = np.array([p1, p2, p3, p4])
        rect = Poly3DCollection([vertices], alpha=0.3, facecolor='gray', edgecolor='none')
        self.ax.add_collection3d(rect)
        
        self.draw()


class LogReader(QThread):
    new_log_signal = pyqtSignal(str, str)
    def __init__(self, device_filter=None, parent=None):
        super(LogReader, self).__init__(parent)
        self.device_filter = device_filter
        self.buffer = []

    def set_device_filter(self, device):
        self.device_filter = device

    def read_logs(self):
        cmd = ["adb", "logcat", "-s", "XRC", "-v", "time"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pattern = re.compile(f"Pose\(RT\)\s+\[0\]\[{self.device_filter}\]:\[([-?\d+.\d+,\s]+?)\],\[([-?\d+.\d+,\s]+?)\]")
        while True:
            line = process.stdout.readline().decode('utf-8')
            match = pattern.search(line)
            if match:
                rvec_str, tvec_str = match.groups()
                rvec = np.array([float(x) for x in rvec_str.split(',')])
                tvec = np.array([float(x) for x in tvec_str.split(',')])

                location, direction = world_location_rotation_from_opencv(rvec, tvec)
                self.buffer.append((line, f"Location: {location}, Direction: {direction}"))
                
                if len(self.buffer) >= 5:
                    for log, parsed_log in self.buffer:
                        self.new_log_signal.emit(log, parsed_log)
                    self.buffer = []
    def run(self):
        self.read_logs()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(500, 500, 1000, 600) # 창 크기 조정
        self.setWindowTitle('Log Viewer')

        mainLayout = QHBoxLayout()

        # 3D Graphics View UI
        self.quiver_view = Quiver3DGraphicsView(self)
        mainLayout.addWidget(self.quiver_view)

        # Log Viewer UI
        logUI = QVBoxLayout()

        self.text_edit = QTextEdit(self)
        logUI.addWidget(self.text_edit)

        exit_btn = QPushButton("Exit", self)
        exit_btn.clicked.connect(self.close)
        logUI.addWidget(exit_btn)

        mainLayout.addLayout(logUI)
        self.setLayout(mainLayout)
        self.show()

    @pyqtSlot(str, str)
    def update_log(self, log_line, processed_data):
        try:
            self.text_edit.append(log_line)
            self.text_edit.append(processed_data)

            location_pattern = r"Location: \[([-?\d+.\d+,\s]+)\]"
            direction_pattern = r"Direction: \[([-?\d+.\d+,\s]+)\]"

            loc_match = re.search(location_pattern, processed_data)
            dir_match = re.search(direction_pattern, processed_data)

            timestamp_pattern = re.compile(r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})")
            timestamp_match = timestamp_pattern.search(log_line)
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                timestamp = datetime.datetime.strptime(timestamp_str, "%m-%d %H:%M:%S.%f")

            if loc_match and dir_match:
                location_str = loc_match.group(1).split()
                direction_str = dir_match.group(1).split()
                location = [float(x) for x in location_str]
                direction = [float(x) for x in direction_str]

                #여기서 location과 direction을 이용하여 그래픽 표현
                # Update 3D Graphics
                self.quiver_view.update_data(location, direction)
    
        except Exception as e:
            print(f"Error in update_log: {e}")                 

if __name__ == '__main__':
    app = QApplication([])
    ex = App()

    log_reader = LogReader()
    log_reader.new_log_signal.connect(ex.update_log)
    log_reader.set_device_filter("1")
    log_reader.start()

    app.exec_()