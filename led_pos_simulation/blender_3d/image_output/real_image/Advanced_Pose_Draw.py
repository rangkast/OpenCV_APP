from Advanced_Function import *

def get_pose_from_adb_log(q):
    cmd = ["adb", "logcat", "-s", "XRC", "-v", "time"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    pattern = re.compile(r"Pose\(RT\)\s+\[0\]\[(\d+)\]:\[([-?\d+.\d+,\s]+)\],\[([-?\d+.\d+,\s]+)\]")

    while not exit_signal.is_set():
        line = process.stdout.readline().decode('utf-8')
        match = pattern.search(line)
        if match:
            index, rvec_str, tvec_str = match.groups()
            rvec = np.array([float(x) for x in rvec_str.split(',')])
            tvec = np.array([float(x) for x in tvec_str.split(',')])
            q.put((index, rvec, tvec))

def on_key(event):
    if event.key == 'q':
        plt.close()

def draw_realtime_quiver(q):
    global exit_signal
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = {'0': 'b', '1': 'r'}
    quivers = {}

    fig.canvas.mpl_connect('key_press_event', on_key)  # q키 감지 이벤트 연결

    plt.show(block=False)  # 창을 비동기로 표시

    while not exit_signal.is_set():
        if not q.empty():
            index, rvec, tvec = q.get()
            location, direction = world_location_rotation_from_opencv(rvec, tvec)
            
            if index in quivers:
                quivers[index].remove()

            quivers[index] = ax.quiver(location[0], location[1], location[2], direction[0], direction[1], direction[2], color=colors.get(index, 'g'), length=1, normalize=True)
            ax.scatter(location[0], location[1], location[2], marker='o')

            ax.set_xlim([-5, 5])
            ax.set_ylim([-5, 5])
            ax.set_zlim([-5, 5])

            plt.draw()
            plt.pause(0.1)

    plt.close(fig)

def keyboard_input_thread():
    input("Press any key to stop...")
    exit_signal.set()

if __name__ == '__main__':
    exit_signal = threading.Event()
    q = Queue()
    input_thread = threading.Thread(target=keyboard_input_thread)
    log_thread = threading.Thread(target=get_pose_from_adb_log, args=(q,))

    input_thread.start()
    log_thread.start()
    draw_realtime_quiver(q)

    input_thread.join()
    log_thread.join()
    plt.close('all')  # 모든 창 닫기
    sys.exit()  # 프로그램 종료
