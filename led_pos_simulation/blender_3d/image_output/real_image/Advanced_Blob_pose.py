
from Advanced_Function import *
import numpy as np
import subprocess
import re
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout, QGroupBox
from PyQt5.QtCore import QTimer, pyqtSignal, pyqtSlot, QThread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import datetime

class LogReader(QThread):
    new_log_signal = pyqtSignal(str, str)

    def run(self):
        cmd = ["adb", "logcat", "-s", "XRC", "-v", "time"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        pattern = re.compile(r"Pose\(RT\)\s+\[0\]\[(\d+)\]:\[([-?\d+.\d+,\s]+)\],\[([-?\d+.\d+,\s]+)\]")

        while True:
            line = process.stdout.readline().decode('utf-8')
            match = pattern.search(line)
            if match:
                index, rvec_str, tvec_str = match.groups()
                rvec = np.array([float(x) for x in rvec_str.split(',')])
                tvec = np.array([float(x) for x in tvec_str.split(',')])
                
                location, direction = world_location_rotation_from_opencv(rvec, tvec)
                print(f"location {location} direction {direction}")
                self.new_log_signal.emit(line, f"Location: {location}, Direction: {direction}")


class Quiver3DPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.canvas = FigureCanvas(Figure())
        layout.addWidget(self.canvas)

        self.ax = self.canvas.figure.add_subplot(111, projection='3d')
        self.canvas.figure.tight_layout()
        self.setLayout(layout)

    def plot_quiver(self, location, direction):
        self.ax.clear()
        self.ax.quiver(*location, *direction, color='b', length=0.5, normalize=True)
        self.ax.scatter(*location, c='k', marker='o', label='')
        self.ax.set_xlim([-1.0, 1.0])
        self.ax.set_ylim([-1.0, 1.0])
        self.ax.set_zlim([-1.0, 1.0])
        self.canvas.draw()

# 실시간 차트를 위한 클래스 추가
class RealTimeChart(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.canvas = FigureCanvas(Figure())
        layout.addWidget(self.canvas)
        
        self.ax = self.canvas.figure.add_subplot(111)
        self.canvas.figure.tight_layout()
        self.data = [0] * 100
        self.counter = 0  # count 변수 추가

        self.setLayout(layout)

    def update_chart(self, new_data_point):
        self.data.append(new_data_point)
        self.data.pop(0)
        
        self.counter += 1  # count 증가
        
        self.ax.clear()
        self.ax.plot(range(self.counter, self.counter + 100), self.data)  # count를 기반으로 x축 데이터 설정
        self.canvas.draw()


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(500, 500, 1000, 500)
        self.setWindowTitle('Log Viewer')

        mainLayout = QHBoxLayout()

        # Log Viewer UI
        logUI = QVBoxLayout()

        self.text_edit = QTextEdit(self)
        logUI.addWidget(self.text_edit)

        exit_btn = QPushButton("Exit", self)
        exit_btn.clicked.connect(self.close)
        logUI.addWidget(exit_btn)

        # Quiver Plot UI
        self.quiver_plot = Quiver3DPlot()
        mainLayout.addLayout(logUI)
        mainLayout.addWidget(self.quiver_plot)

        # ECG-like Plot UI (placeholder for now)
        ecgUI = QGroupBox("ECG-like Data Plot")
        mainLayout.addWidget(ecgUI)

        # 실시간 차트 UI 추가
        self.real_time_chart = RealTimeChart()
        mainLayout.addWidget(self.real_time_chart)
        
        # Timer 설정
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_real_time_chart)
        self.timer.start(10)  # 10ms마다 차트 업데이트

        self.setLayout(mainLayout)
        self.show()

    @pyqtSlot(str, str)
    def update_log(self, log_line, processed_data):
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
            self.quiver_plot.plot_quiver(location, direction)
            self.real_time_chart.update_chart(location[0])  # x 값만 사용
            
    def update_real_time_chart(self):
        # 주의: 이 함수는 현재 사용되지 않지만 필요한 경우를 위해 남겨둡니다.
        # 실제로는 update_log 함수에서 실시간 차트를 업데이트하고 있습니다.
        pass  # 필요하지 않은 경우 이 함수를 제거할 수 있습니다.         

if __name__ == '__main__':
    app = QApplication([])
    ex = App()

    log_reader = LogReader()
    log_reader.new_log_signal.connect(ex.update_log)
    log_reader.start()

    app.exec_()