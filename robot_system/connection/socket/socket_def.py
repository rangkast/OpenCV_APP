# 접속할 서버 주소, 여기에는 루프백(loopback) 인터페이스 주소 즉 localhost를 사용
# HOST = '192.168.0.2'
HOST = '127.0.0.1'

# 클라이언트 접속을 대기하는 포트 번호
# PORT = 4444
PORT = 8808

UDP_MODE = 1
TCP_MODE = 0

class TVEC_INDEX:
    x = 0
    y = 1
    z = 2

class RVEC_INDEX:
    Rall = 0
    Pitch = 1
    Yaw = 2
    
class RECEIVE_DATA:
    def __init__(self) :
        self.t_vec = []
        self.r_vec = []
        self.status = 'ok'

current_t = [30, 30, 30]
current_r = [0, 0, 0]

rt_cmd = ['x','y','z','R','P','Y']
rt_list = []
rt_val_list = []

def cmd_check(cmd_list):
    rt_list.clear()
    rt_val_list.clear()

    if len(cmd_list) % 2 != 0:
        return ("[InvalidCommand] : The number of commands is wrong")
    else:
        for idx, cmd in enumerate(cmd_list):
            if idx % 2 == 0:
                if cmd not in rt_cmd :
                    return ("[InvalidCommand] : rt command does not exist")
                else :
                    rt_list.append(cmd)
            else :
                try :
                    rt_val_list.append(float(cmd))
                except :
                    return ("[InvalidCommand] : rt_value is not float")
    return "Success"

def recv_msg_from_server(recv_data):
    r_data = RECEIVE_DATA()
    recv_msg = str(recv_data.decode('utf-8')).split()

    if recv_msg[0] != "[InvalidCommand]" :
        for t_val in recv_msg[3:6]:
            r_data.t_vec.append(float(t_val))
        for r_val in recv_msg[8:11]:
            r_data.r_vec.append(float(r_val))
    else :
        r_data.stats = 'fail'
    return r_data

def trans_cmd_func(sock, cmd, addr):
    if TCP_MODE == 1 :
        sock.sendall(cmd.encode('utf-8'))
    else : #UDP Mode
        sock.sendto(cmd.encode('utf-8'), addr)

def recv_cmd_func(sock):
    if TCP_MODE == 1:
        addr = 0
        # 포트 사용중이라 연결할 수 없다는
        # WinError 10048 에러 해결을 위해 필요
        recv_data = sock.recv(1024)
    else : #UDP Mode
        recv_data, addr = sock.recvfrom(1024)
    return recv_data, addr