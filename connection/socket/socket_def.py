import socket
import sys


# 접속할 서버 주소,여기에는 루프백(loopback) 인터페이스 주소 즉 localhost를 사용
# HOST = '192.168.0.2'
# HOST = '127.0.0.1'

HOST = '192.168.1.10'
RECV_HOST = '192.168.1.2'

# 클라이언트 접속을 대기하는 포트 번호
# PORT = 8808
PORT = 61559
RECV_PORT = 61554

UDP_MODE = 1
TCP_MODE = 0

DEBUG_PRINT = 0

trsb = 2
rvsb = 3

start_cmd = chr(4)
end_cmd = chr(5)
set_cmd = chr(1)
ctr_cmd = chr(2)
read_cmd = chr(3)
retrun_cmd = chr(10)

read_cmd_list = ['cprr','cpar','cpj','cpt','cps','sys_set','s_dt']

cpj = start_cmd +','+ read_cmd +',cpj,'+ end_cmd
cpar = start_cmd +','+ read_cmd +',cpar,'+ end_cmd
cpt = start_cmd +','+ read_cmd +',cpt,'+ end_cmd
cps = start_cmd +','+ read_cmd +',cps,'+ end_cmd
cprr = start_cmd +','+ read_cmd +',cprr,'+ end_cmd
sys_set = start_cmd +','+ read_cmd +',sys_set,'+ end_cmd
s_dt = start_cmd +','+ read_cmd +',s_dt,'+ end_cmd

robot_coord = cpar

class Setting_CMD:
    ar_coord = 'rc'
    offset = {'x':0,'y':0,'z':0,'a':0,'b':0,'c':0}
    mv_sp = 10         # msec
    save_rb = 'y'       # y or n
    save_itv = 100       # meec
    ts = 'on'           # on or off
    ss = 'off'          # on or off
    
    # cmd_line = '{},{},{},ox:{},oy:{},oz:{},oa:{},ob:{},oc:{},mv_sp:{},save_rb:{},save_itv:{},{}'.format(
    #     start_cmd,set_cmd,ar_coord,offset['x'],offset['y'],offset['z'],offset['a'],offset['b'],offset['c'],mv_sp,
    #     save_rb,save_itv,end_cmd)

    # def update_cmd_line(self):
    #     self.cmd_line = '{},{},{},ox:{},oy:{},oz:{},oa:{},ob:{},oc:{},mv_sp:{},save_rb:{},save_itv:{},{}'.format(
    #     start_cmd,set_cmd,self.ar_coord,self.offset['x'],self.offset['y'],self.offset['z'],self.offset['a'],self.offset['b'],
    #     self.offset['c'],self.mv_sp,self.save_rb,self.save_itv,end_cmd)
    
    def trans_setting(self):
        cmd_line = '{},{},{},ox:{:.2f},oy:{:.2f},oz:{:.2f},oa:{:.2f},ob:{:.2f},oc:{:.2f},mv_sp:{},save_rb:{},save_itv:{},{}'.format(
            start_cmd,set_cmd,self.ar_coord,self.offset['x'],self.offset['y'],self.offset['z'],self.offset['a'],
            self.offset['b'],self.offset['c'],self.mv_sp,self.save_rb,self.save_itv,end_cmd)
        recv_msg = send_cmd_to_server(cmd_line)
        check_data(cmd_line,recv_msg,sys_set)

    def set_data(self,data):
        self.ar_coord = data[0]
        self.offset = {'x':data[1].split(':')[1],'y':data[2].split(':')[1],'z':data[3].split(':')[1],
                       'a':data[4].split(':')[1],'b':data[5].split(':')[1],'c':data[6].split(':')[1]}
        self.mv_sp = data[7].split(':')[1]
        self.save_rb = data[8].split(':')[1]
        self.save_itv = data[9].split(':')[1]
        # self.ts = data[10].split(':')[1]
        # self.ss = data[11].split(':')[1]
            
    def show_data(self):
        print('ar_coord:{},ox:{},oy:{},oz:{},oa:{},ob:{},oc:{},mv_sp:{},save_rb:{},save_itv:{},ts:{},ss:{}'.format(
            self.ar_coord,self.offset['x'],self.offset['y'],self.offset['z'],self.offset['a'],self.offset['b'],self.offset['c'],
            self.mv_sp,self.save_rb,self.save_itv,self.ts,self.ss))

class Controll_CMD:
    joint = {'1':0,'2':0,'3':0,'4':0,'5':0,'6':0}
    ar_pos = {'x':0,'y':0,'z':0,'a':0,'b':0,'c':0}
    rr_pos = {'x':0,'y':0,'z':0,'a':0,'b':0,'c':0}
    tb_pos = {'x':0,'y':0,'z':0}
    sb_pos = {'y':0,'z':0}
    ts_dis = -1
    ss_dis = -1
    
#     cmd_line = "{},{},j1:{},j2:{},j3:{},j4:{},j5:{},j6:{},\
# rx:{},ry:{},rz:{},ra:{},rb:{},rc:{},tx:{},ty:{},tz:{},sx:{},sy:{},ts:{},ss:{}".format(
#         start_cmd,ctr_cmd,joint['1'],joint['2'],joint['3'],joint['4'],joint['5'],joint['6'],
#                          ar_pos['x'],ar_pos['y'],ar_pos['z'],ar_pos['a'],ar_pos['b'],ar_pos['c'],
#                          tb_pos['x'],tb_pos['y'],tb_pos['z'],sb_pos['y'],sb_pos['z'],us,ss,end_cmd)

#     def update_cmd_line(self):
#         self.cmd_line = "{},{},j1:{},j2:{},j3:{},j4:{},j5:{},j6:{},\
# rx:{},ry:{},rz:{},ra:{},rb:{},rc:{},tx:{},ty:{},tz:{},sx:{},sy:{},ts:{},ss:{}".format(
#                 start_cmd,ctr_cmd,self.joint['1'],self.joint['2'],self.joint['3'],self.joint['4'],
#                          self.joint['5'],self.joint['6'],self.r_pos['x'],self.r_pos['y'],self.r_pos['z'],
#                          self.r_pos['a'],self.r_pos['b'],self.r_pos['c'],self.tb_pos['x'],self.tb_pos['y'],
#                          self.tb_pos['z'],self.sb_pos['x'],self.sb_pos['y'],self.us,self.ss,end_cmd)

    def set_joint_data(self,data):
        self.joint = {'1':data[0].split(':')[1],'2':data[1].split(':')[1],'3':data[2].split(':')[1],
                       '4':data[3].split(':')[1],'5':data[4].split(':')[1],'6':data[5].split(':')[1]}
        
    def set_arpos_data(self,data):
        self.ar_pos = {'x':data[0].split(':')[1],'y':data[1].split(':')[1],'z':data[2].split(':')[1],
                       'a':data[3].split(':')[1],'b':data[4].split(':')[1],'c':data[5].split(':')[1]}
    
    def set_rrpos_data(self,data):
        self.rr_pos = {'x':data[0].split(':')[1],'y':data[1].split(':')[1],'z':data[2].split(':')[1],
                       'a':data[3].split(':')[1],'b':data[4].split(':')[1],'c':data[5].split(':')[1]}
    
    def set_tbpos_data(self,data):
        self.tb_pos = {'x':data[0].split(':')[1],'y':data[1].split(':')[1],'z':data[2].split(':')[1]}
    
    def set_sbpos_data(self,data):
        self.sb_pos = {'y':data[0].split(':')[1],'z':data[1].split(':')[1]}
    
    def set_ss_data(self,data):
        self.ts_dis = data[0]
        self.ss_dis= data[1]

    def show_data(self):
        print('- joint angle \n{} \n- ar robot_position \n{} \n- rr robot_position \n{} \n- top_bar_position & distance\
            \n{} dis : {} \n- side_bar_position & distance\n{} dis : {}'.format(
            self.joint,self.ar_pos,self.rr_pos,self.tb_pos,self.ts_dis,self.sb_pos,self.ss_dis
        ))

class RB_INFO:
    st = Setting_CMD()
    ctr = Controll_CMD()
    
    def show_all_data(self):
        print("# Setting Value")
        self.st.show_data()
        print("# Controll Value")
        self.ctr.show_data()
        
    def sync_robot(self):
        for rd_cmd in read_cmd_list:
            rd_cmd_form = start_cmd +','+ read_cmd +',' + rd_cmd +','+ end_cmd
            recv = send_cmd_to_server(rd_cmd_form)
            update_rb_info(rd_cmd,recv[2:])
            if rd_cmd == 'sys_set':
                self.st.ts = recv[-3].split(':')[1]
                self.st.ss = recv[-2].split(':')[1]

rb_info = RB_INFO()

def trans_joint(data):
        cmd_line = '{},{},j1:{:.2f},j2:{:.2f},j3:{:.2f},j4:{:.2f},j5:{:.2f},j6:{:.2f},{}'.format(
            start_cmd,ctr_cmd,data['1'],data['2'],data['3'],data['4'],data['5'],data['6'],end_cmd)
        if rb_info.st.ar_coord == 'ac':
            recv_msg = send_cmd_to_server(cmd_line)
            check_data(cmd_line,recv_msg,cpj)
        elif rb_info.st.ar_coord == 'rc':
            before_tbpos = send_cmd_to_server(cpj)
            # before_tbpos = ['\x04', '\n', 'cpj','j1:0.00','j2:1.00','j3:0.00','j4:0.00','j5:1.00','j6:0.00','0x05']
            recv_msg = send_cmd_to_server(cmd_line)
            if recv_msg[2] == 'Qok' :
                if DEBUG_PRINT > 1 :
                    print("Robot Opeartion Sucess")
                after_tbpos = send_cmd_to_server(cpj)
                # after_tbpos = ['\x04', '\n', 'cpj','j1:2','j2:5.00','j3:5.00','j4:1.00','j5:3.00','j6:3.0','0x05']
                temp_data = []
                for i,after_data in enumerate(after_tbpos[rvsb:-1]):
                    temp_data.append(float(after_data.split(':')[1]) - float(before_tbpos[i+rvsb].split(':')[1]))
            
                ab_diff_cmd = '{},{},j1:{:.2f},j2:{:.2f},j3:{:.2f},j4:{:.2f},j5:{:.2f},j6:{:.2f},{}'.format(
                start_cmd,ctr_cmd,temp_data[0],temp_data[1],temp_data[2],temp_data[3],temp_data[4],temp_data[5],end_cmd)
                if cmd_line == ab_diff_cmd :
                    if DEBUG_PRINT > 1 :
                        print("Controll Data Check Sucess")
                    update_rb_info('cpj',after_tbpos[rvsb:])
                else :
                        print("!!!!Controll Data Check Fail!!!!")
            else:
                print("!!!!Robot Opeartion Fail!!!!")  
        
def trans_rpos(data):
        cmd_line = '{},{},rx:{:.2f},ry:{:.2f},rz:{:.2f},ra:{:.2f},rb:{:.2f},rc:{:.2f},{}'.format(
            start_cmd,ctr_cmd,data['x'],data['y'],data['z'],data['a'],data['b'],data['c'],end_cmd)
        if rb_info.st.ar_coord == 'ac':
            recv_msg = send_cmd_to_server(cmd_line)
            check_data(cmd_line,recv_msg,robot_coord)
        elif rb_info.st.ar_coord == 'rc':
            before_tbpos = send_cmd_to_server(robot_coord)
            # before_tbpos = ['\x04', '\n', 'cpar','rx:0.00','ry:1.00','rz:0.00','ra:0.00','rb:1.00','rc:1.00','\x05']
            recv_msg = send_cmd_to_server(cmd_line)
            if recv_msg[2] == 'Qok' :
                if DEBUG_PRINT > 1 :
                    print("Robot Opeartion Sucess")
                after_tbpos = send_cmd_to_server(robot_coord)
                # after_tbpos = ['\x04', '\n', 'cpar','rx:1','ry:1','rz:0.00','ra:0','rb:1','rc:1','\x05']
                temp_data = []
                for i,after_data in enumerate(after_tbpos[rvsb:-1]):
                    temp_data.append(float(after_data.split(':')[1]) - float(before_tbpos[i+rvsb].split(':')[1]))
            
                ab_diff_cmd = '{},{},rx:{:.2f},ry:{:.2f},rz:{:.2f},ra:{:.2f},rb:{:.2f},rc:{:.2f},{}'.format(
                start_cmd,ctr_cmd,temp_data[0],temp_data[1],temp_data[2],temp_data[3],temp_data[4],temp_data[5],end_cmd)
                if cmd_line == ab_diff_cmd :
                    if DEBUG_PRINT > 1 :
                        print("Controll Data Check Sucess")
                    if robot_coord == cprr:
                        update_rb_info('cprr',after_tbpos[rvsb:])
                    elif robot_coord == cpar:
                        update_rb_info('cpar',after_tbpos[rvsb:])
                else :
                        print("!!!!Controll Data Check Fail!!!!")
            else:
                print("!!!!Robot Opeartion Fail!!!!")

def trans_tbpos(data):
        cmd_line = '{},{},tx:{:.2f},ty:{:.2f},tz:{:.2f},{}'.format(
            start_cmd,ctr_cmd,data['x'],data['y'],data['z'],end_cmd)
        if rb_info.st.ar_coord == 'ac':
            recv_msg = send_cmd_to_server(cmd_line)
            check_data(cmd_line,recv_msg,cpt)
        elif rb_info.st.ar_coord == 'rc':
            before_tbpos = send_cmd_to_server(cpt)
            # before_tbpos = ['\x04', '\n', 'cpt','tx:0.00','ty:1.00','tz:0.00','0x05']
            recv_msg = send_cmd_to_server(cmd_line)
            if recv_msg[2] == 'Qok' :
                if DEBUG_PRINT > 1 :
                    print("Robot Opeartion Sucess")
                after_tbpos = send_cmd_to_server(cpt)
                # after_tbpos = ['\x04', '\n', 'cpt','tx:1','ty:3.3','tz:4','0x05']
                temp_data = []
                for i,after_data in enumerate(after_tbpos[rvsb:-1]):
                    temp_data.append(float(after_data.split(':')[1]) - float(before_tbpos[i+rvsb].split(':')[1]))
            
                ab_diff_cmd = '{},{},tx:{:.2f},ty:{:.2f},tz:{:.2f},{}'.format(
                start_cmd,ctr_cmd,temp_data[0],temp_data[1],temp_data[2],end_cmd)
                if cmd_line == ab_diff_cmd :
                    if DEBUG_PRINT > 1 :
                        print("Controll Data Check Sucess")
                    update_rb_info('cpt',after_tbpos[rvsb:])
                else :
                        print("!!!!Controll Data Check Fail!!!!")
            else:
                print("!!!!Robot Opeartion Fail!!!!")                

def trans_sbpos(data):
        cmd_line = '{},{},sy:{:.2f},sz:{:.2f},{}'.format(
            start_cmd,ctr_cmd,data['y'],data['z'],end_cmd)
        if rb_info.st.ar_coord == 'ac':
            recv_msg = send_cmd_to_server(cmd_line)
            check_data(cmd_line,recv_msg,cps)
        elif rb_info.st.ar_coord == 'rc':
            before_tbpos = send_cmd_to_server(cps)
            # before_tbpos = ['\x04', '\n', 'cps','sy:1.00','sz:1.00','0x05']
            recv_msg = send_cmd_to_server(cmd_line)
            if recv_msg[2] == 'Qok' :
                if DEBUG_PRINT > 1 :
                    print("Robot Opeartion Sucess")
                after_tbpos = send_cmd_to_server(cps)
                # after_tbpos = ['\x04', '\n', 'cps','sy:1','sz:2.11','0x05']
                temp_data = []
                for i,after_data in enumerate(after_tbpos[rvsb:-1]):
                    temp_data.append(float(after_data.split(':')[1]) - float(before_tbpos[i+rvsb].split(':')[1]))
            
                ab_diff_cmd = '{},{},sy:{:.2f},sz:{:.2f},{}'.format(
                start_cmd,ctr_cmd,temp_data[0],temp_data[1],end_cmd)
                if cmd_line == ab_diff_cmd :
                    if DEBUG_PRINT > 1 :
                        print("Controll Data Check Sucess")
                    update_rb_info('cps',after_tbpos[rvsb:])
                else :
                        print("!!!!Controll Data Check Fail!!!!")
            else:
                print("!!!!Robot Opeartion Fail!!!!")                

        
def trans_ts(data):
        cmd_line = '{},{},ts:{},{}'.format(
            start_cmd,ctr_cmd,data,end_cmd)
        sp_cmd = cmd_line.replace(" ","").split(',')
        recv_msg = send_cmd_to_server(cmd_line)
        if recv_msg[2] =='Qok':
            rv_data = send_cmd_to_server(sys_set)
            if sp_cmd[2] == rv_data[-3]:
                rb_info.st.ts = sp_cmd[2].split(':')[1]
            else:
                if DEBUG_PRINT > 1 :
                    print('!!!!top sensor data check fail!!!!')
        else:
            print("!!!!Top Bar Operation Fail!!!!")
                
def trans_ss(data):
        cmd_line = '{},{},ss:{},{}'.format(
            start_cmd,ctr_cmd,data,end_cmd)
        sp_cmd = cmd_line.replace(" ","").split(',')
        recv_msg = send_cmd_to_server(cmd_line)
        if recv_msg[2] =='Qok':
            rv_data = send_cmd_to_server(sys_set)
            if sp_cmd[2] == rv_data[-2]:
                rb_info.st.ss = sp_cmd[2].split(':')[1]
            else:
                if DEBUG_PRINT > 1 :
                    print('!!!!top sensor data check fail!!!!')
        else:
            print("!!!!Top Bar Operation Fail!!!!")

def change_decimal_point(rv_data):
    for i,data in enumerate(rv_data):
        rv_data[i] = data.split(':')[0] +":"+ format(float(data.split(':')[1]),".2f")   
    return rv_data


def update_rb_info(rd_cmd,recv_msg):
    if rd_cmd in read_cmd_list:
        if rd_cmd == 'cpar':
            rb_info.ctr.set_arpos_data(recv_msg)
        elif rd_cmd == 'cprr':
            rb_info.ctr.set_rrpos_data(recv_msg)
        elif rd_cmd == 'cpj':
            rb_info.ctr.set_joint_data(recv_msg)
        elif rd_cmd == 'cpt':
            rb_info.ctr.set_tbpos_data(recv_msg)
        elif rd_cmd == 'cps':
            rb_info.ctr.set_sbpos_data(recv_msg)
        elif rd_cmd == 'sys_set':
            rb_info.st.set_data(recv_msg)
        elif rd_cmd == 's_dt':
            rb_info.ctr.set_ss_data(recv_msg)
    else :
        print("A Read Command named '{}' does not exist.".format(rd_cmd))

def check_data(cmd,recv_msg,attr):
    sp_cmd = cmd.replace(" ","").split(',')
    if recv_msg[2] == 'Qok':
        if DEBUG_PRINT > 1 :
            print("Robot Opeartion Sucess")
        if sp_cmd[1] == set_cmd:
            rv_data = send_cmd_to_server(attr)
            if (sp_cmd[trsb] == rv_data[rvsb]) & (change_decimal_point(sp_cmd[trsb+1:10]) == change_decimal_point(rv_data[rvsb+1:11])):   # offset check
                if sp_cmd[8:-1] == rv_data[9:-3]:
                    if DEBUG_PRINT > 1 :
                        print("Setting Data Check Sucess")
                    rd_cmd = attr.replace(" ","").split(',')[2]
                    update_rb_info(rd_cmd,rv_data[rvsb:])
                else:
                    print("!!!!#2 Setting Data Check Fail!!!!")
            else:
                print("!!!!#1 Setting Data Check Fail!!!!")
        elif sp_cmd[1] == ctr_cmd:
            rv_data = send_cmd_to_server(attr)
            if change_decimal_point(sp_cmd[trsb:-1]) == change_decimal_point(rv_data[rvsb:-1]):
                if DEBUG_PRINT > 1 :
                    print("Controll Data Check Sucess")
                rd_cmd = attr.replace(" ","").split(',')[2]
                update_rb_info(rd_cmd,rv_data[rvsb:])
            else:
                print("!!!!Controll Data Check Fail!!!!")
    else :
        print("!!!!Robot Opeartion Fail!!!!")

def send_cmd_to_server(cmd):
    # connect server
    if TCP_MODE == 1 :
        server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        server_socket.connect((HOST,PORT))
    else : #UDP MODE
        server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        rec_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        rec_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        rec_socket.bind((RECV_HOST,RECV_PORT))

    # trans cmd
    trans_cmd_func(server_socket,cmd,(HOST,PORT))
    
    # receive data
    recv_data,addr = recv_cmd_func(rec_socket)
    # recv_data,addr = recv_cmd_func(server_socket)
    recv_msg = recv_msg_from_server(recv_data)
    
    server_socket.close()

    return recv_msg

def recv_msg_from_server(recv_data):
    recv_data = recv_data.decode('utf-8')
    sp_cmd = recv_data.replace(" ","").split(',')

    return sp_cmd

def trans_cmd_func(sock,cmd,addr):
    if DEBUG_PRINT > 0 :
        print("Trans_CMD >> ",cmd)
    if TCP_MODE == 1 :
        sock.sendall(cmd.encode('utf-8'))
    else : #UDP Mode
        encded = cmd.encode('utf-8')
        sock.sendto(encded,addr)
        # print(f'{cmd}:{encded}')
        # sock.sendto(cmd,addr)

def recv_cmd_func(sock):
    if TCP_MODE == 1:
        addr = 0
        # 포트 사용중이라 연결할 수 없다는
        # WinError 10048 에러 해결을 위해 필요
        recv_data = sock.recv(1024)
    else : #UDP Mode
        recv_data,addr = sock.recvfrom(1024)
    if DEBUG_PRINT > 0 :
        print("Recv_CMD >> ",repr(recv_data.decode('utf-8')))
    return recv_data,addr


def socket_cmd_to_robot(target, ar_coord, cmd):
    # setting cmd -> offset 만 변경 시
    test_set = Setting_CMD()
    MV_SPEED = 100
    if ar_coord == 'ac':
        test_set.ar_coord = 'ac'
        test_set.mv_sp = MV_SPEED
    else:
        test_set.ar_coord = 'rc'
        test_set.mv_sp = MV_SPEED
    test_set.trans_setting()

    if target == 'tbpos':
        trans_tbpos(cmd)
    elif target == 'joint':
        trans_joint(cmd)
    elif target == 'sbpos':
        trans_sbpos(cmd)

    send_cmd_to_server(sys_set)
    # rb_info.show_all_data()
