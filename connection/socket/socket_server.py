import select
import _thread

# sys.path.append('/home/hs13.yang/ws_rb_interface')

from socket_def import *

def cmd_process_func(client_socket,recv_cmd,addr):
    if 0:
        global rt_list,rt_val_list,current_t,current_r
        # 현재 로봇 팔 위치 전송
        if (recv_cmd[0] == 'CL'):
            return_msg = "[CurrentLocation] tvec : " + " ".join(map(str,current_t)) +" rvec : "+" ".join(map(str,current_r))
            trans_cmd_func(client_socket,return_msg,addr)

        # Bar와 로봇팔 위치를 초기화
        elif (recv_cmd[0] == 'RS'):
            if reset_location() == "ROSuccess":
                return_msg = "[ResetLocation] tvec : " + " ".join(map(str,current_t)) +" rvec : "+" ".join(map(str,current_r))
            trans_cmd_func(client_socket,return_msg,addr)

        elif (recv_cmd[0] == 'HR'):
            recv_cmd = recv_cmd[1:]

            rt_list.clear()
            rt_val_list.clear()
            rt_list.append('Y')

            try :
                rt_val_list.append(float(recv_cmd[0]))
            except :
                return_msg = "[InvalidCommand] : angle is not float"

            if len(rt_list) == len(rt_val_list):
                if robot_operation() == "ROSuccess":
                    # print(" ".join(map(str,current_t)))
                    # print(" ".join(map(str,current_r)))
                    return_msg = "[OPSuccess] tvec : " + " ".join(map(str,current_t)) +" rvec : "+" ".join(map(str,current_r))
                else:
                    return_msg = "[OPFail] tvec : " + " ".join(map(str,current_t)) +" rvec : "+" ".join(map(str,current_r))

            trans_cmd_func(client_socket,return_msg,addr)

        elif (recv_cmd[0] == 'FC'):
            recv_cmd = recv_cmd[1:]
            return_msg = cmd_check(recv_cmd)

            if return_msg == "Success":
                if len(rt_list) == len(rt_val_list):
                    if robot_operation() == "ROSuccess":
                        # print(" ".join(map(str,current_t)))
                        # print(" ".join(map(str,current_r)))
                        return_msg = "[OPSuccess] tvec : " + " ".join(map(str,current_t)) +" rvec : "+" ".join(map(str,current_r))
                    else:
                        return_msg = "[OPFail] tvec : " + " ".join(map(str,current_t)) +" rvec : "+" ".join(map(str,current_r))

            trans_cmd_func(client_socket,return_msg,addr)

    else :
        print('recv_cmd:',recv_cmd)
        print(recv_cmd[2])
        if recv_cmd[2].split(',')[0] == 'cpj':
            print('cpj')
            trans_cmd_func(client_socket,start_cmd+','+retrun_cmd+','+recv_cmd[2].split(',')[0]+','+"j1:2,j2:4,j3:5,j4:1,j5:2,j6:3"+','+end_cmd,addr)
        elif recv_cmd[2].split(',')[0] == 'cpar':
            print('cpar')
            trans_cmd_func(client_socket,start_cmd+','+retrun_cmd+','+recv_cmd[2].split(',')[0]+','+"rx:1,ry:0,rz:0,ra:0,rb:0,rc:1"+','+end_cmd,addr)
        elif recv_cmd[2].split(',')[0] == 'cpt':
            print('cpt')
            trans_cmd_func(client_socket,start_cmd+','+retrun_cmd+','+recv_cmd[2].split(',')[0]+','+"tx:1,ty:2.3,tz:4"+','+end_cmd,addr)
        elif recv_cmd[2].split(',')[0] ==  'cps':
            print('cps')
            trans_cmd_func(client_socket,start_cmd+','+retrun_cmd+','+recv_cmd[2].split(',')[0]+','+"sy:0,sz:1.11"+','+end_cmd,addr)
        elif recv_cmd[2].split(',')[0] ==  'cprj':
            print('cprj')
            trans_cmd_func(client_socket,start_cmd+','+retrun_cmd+','+"j1:0,j2:0,j3:1,j4:0,j5:1,j6:0"+','+end_cmd,addr)
        elif recv_cmd[2].split(',')[0] == 'cprr':
            print('cprr')
            trans_cmd_func(client_socket,start_cmd+','+retrun_cmd+','+recv_cmd[2].split(',')[0]+','+"rx:1,ry:0,rz:0,ra:0,rb:0,rc:1"+','+end_cmd,addr)
        elif recv_cmd[2].split(',')[0] == 'cprt':
            print('cprt')
            trans_cmd_func(client_socket,start_cmd+','+retrun_cmd+','+"tx:0,ty:1,tz:0"+','+end_cmd,addr)
        elif recv_cmd[2].split(',')[0] == 'cprs':
            print('cprs')
            trans_cmd_func(client_socket,start_cmd+','+retrun_cmd+','+"sx:1,sy:0"+','+end_cmd,addr)
        elif recv_cmd[2].split(',')[0] == 'sys_set':
            print('sys_set')
            trans_cmd_func(client_socket,start_cmd+','+retrun_cmd+','+recv_cmd[2].split(',')[0]+','+"ac,ox:23.34,oy:40.00,oz:42.30,oa:5.00,ob:7.00,oc:1.00,mv_sp:10,save_rb:y,save_itv:100.3,ts:on,ss:off"+','+end_cmd,addr)
        elif recv_cmd[2].split(',')[0] == 's_dt':
            print('s_dt')
            trans_cmd_func(client_socket,start_cmd+','+retrun_cmd+','+recv_cmd[2].split(',')[0]+','+"t:23,s:-1"+','+end_cmd,addr)
        else :
            trans_cmd_func(client_socket,start_cmd+','+retrun_cmd+','+'Qok,'+end_cmd,addr)
            # trans_cmd_func(client_socket,start_cmd+','+set_cmd+','+'QOK,'+end_cmd,addr)

if TCP_MODE == 1 :
    server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    server_socket.bind((HOST,PORT))
    server_socket.listen()
    socketList = [server_socket]

    print("TCP Server Start")

    # select 함수의 timeout 설정
    timeout = 1

    while True:
        try:
            read_socket,write_socket,error_socket = select.select(socketList,[],[],timeout)

            for sock in read_socket:
                if sock == server_socket:
                    client_socket,addr = server_socket.accept()
                    _thread.start_new_thread(tcp_threaded,(client_socket,addr))
                else:
                    message = recv_cmd_func(sock)
                    sock.send(message.decode().upper().encode())
                    sock.close()
                    socketList.remove(sock)

        except KeyboardInterrupt:
            print("Server Shutdown")
            server_socket.close()
            sys.exit()


else : #UDP MODE
    server_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    server_socket.bind((HOST,PORT))
    socketList = [server_socket]

    print("UDP Server Start")

    timeout = 1

    while True:
        try:
            read_socket,write_socket,error_socket = select.select(socketList,[],[],timeout)

            for sock in read_socket:
                if sock == server_socket:
                    data,addr = recv_cmd_func(server_socket)
                    print('Received from' ,addr,data.decode('utf-8'))

                    if not data:
                        print('Disconnected by' + addr[0],':',addr[1])
                        break

                    recv_cmd = str(data.decode('utf-8')).split(',')
                    print("received cmd : ",recv_cmd)

                    cmd_process_func(server_socket,recv_cmd,addr)

        except KeyboardInterrupt:
            print("Server Shutdown")
            server_socket.close()
            sys.exit()