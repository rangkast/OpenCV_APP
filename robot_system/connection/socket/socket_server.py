import socket
import select
import sys
import _thread

from robot_system.connection.socket.socket_def import *
from robot_system.connection.socket.socket_robot import *

def cmd_process_func(client_socket, recv_cmd, addr):
    global rt_list, rt_val_list, current_t, current_r
    # 현재 로봇 팔 위치 전송
    if (recv_cmd[0] == 'CL'):
        return_msg = "[CurrentLocation] tvec : " + " ".join(map(str, current_t)) +" rvec : "+" ".join(map(str, current_r))
        trans_cmd_func(client_socket, return_msg, addr)

    # Bar와 로봇팔 위치를 초기화
    elif (recv_cmd[0] == 'RS'):
        if reset_location() == "ROSuccess":
            return_msg = "[ResetLocation] tvec : " + " ".join(map(str, current_t)) +" rvec : "+" ".join(map(str, current_r))
        trans_cmd_func(client_socket, return_msg, addr)

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
                # print(" ".join(map(str, current_t)))
                # print(" ".join(map(str, current_r)))
                return_msg = "[OPSuccess] tvec : " + " ".join(map(str, current_t)) +" rvec : "+" ".join(map(str, current_r))
            else:
                return_msg = "[OPFail] tvec : " + " ".join(map(str, current_t)) +" rvec : "+" ".join(map(str, current_r))

        trans_cmd_func(client_socket, return_msg, addr)

    elif (recv_cmd[0] == 'FC'):
        recv_cmd = recv_cmd[1:]
        return_msg = cmd_check(recv_cmd)

        if return_msg == "Success":
            if len(rt_list) == len(rt_val_list):
                if robot_operation() == "ROSuccess":
                    # print(" ".join(map(str, current_t)))
                    # print(" ".join(map(str, current_r)))
                    return_msg = "[OPSuccess] tvec : " + " ".join(map(str, current_t)) +" rvec : "+" ".join(map(str, current_r))
                else:
                    return_msg = "[OPFail] tvec : " + " ".join(map(str, current_t)) +" rvec : "+" ".join(map(str, current_r))

        trans_cmd_func(client_socket, return_msg, addr)

    else :
        return_msg = "[InvalidCommand] : command does not exist"
        trans_cmd_func(client_socket, return_msg, addr)


def tcp_threaded(client_socket, addr):

    print("Connected by", addr)

    while True:
        try:
            data, _ = recv_cmd_func(client_socket)
            # print('Received from' ,addr, data.decode('utf-8'))

            if not data:
                print('Disconnected by' + addr[0],':',addr[1])
                break

            recv_cmd = str(data.decode('utf-8')).split()
            print("received cmd : ",recv_cmd)

            cmd_process_func(client_socket, recv_cmd, addr)

        except ConnectionResetError as e:
            print('Disconnected by ' + addr[0],':',addr[1])
            break
    client_socket.close()

if TCP_MODE == 1 :
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    socketList = [server_socket]

    print("TCP Server Start")

    # select 함수의 timeout 설정
    timeout = 1

    while True:
        try:
            read_socket, write_socket, error_socket = select.select(socketList, [], [], timeout)

            for sock in read_socket:
                if sock == server_socket:
                    client_socket, addr = server_socket.accept()
                    _thread.start_new_thread(tcp_threaded, (client_socket, addr))
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
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    socketList = [server_socket]

    print("UDP Server Start")

    timeout = 1

    while True:
        try:
            read_socket, write_socket, error_socket = select.select(socketList, [], [], timeout)

            for sock in read_socket:
                if sock == server_socket:
                    data, addr = recv_cmd_func(server_socket)
                    print('Received from' ,addr, data.decode('utf-8'))

                    if not data:
                        print('Disconnected by' + addr[0],':',addr[1])
                        break

                    recv_cmd = str(data.decode('utf-8')).split()
                    print("received cmd : ",recv_cmd)

                    cmd_process_func(server_socket, recv_cmd, addr)

        except KeyboardInterrupt:
            print("Server Shutdown")
            server_socket.close()
            sys.exit()