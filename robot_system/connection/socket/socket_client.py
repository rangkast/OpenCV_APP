import socket
import sys

from robot_system.connection.socket.socket_def import *

def send_cmd_to_server(cmd):
    # connect server
    if TCP_MODE == 1 :
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.connect((HOST, PORT))
    else : #UDP MODE
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # trans cmd
    trans_cmd_func(server_socket, cmd, (HOST, PORT))
    
    # receive data
    recv_data, addr = recv_cmd_func(server_socket)
    print("Server >> ", repr(recv_data.decode('utf-8')))
    recv_msg = recv_msg_from_server(recv_data)
    
    server_socket.close()

if 0:
    while True:
        try:
            print("==  Input Command (Help->h) ==")
            input_cmd = input("input : ")
            if input_cmd == 'h':
                print("-------------------------------------------HELP-------------------------------------------")
                print("Robot Interface Protocol Commands")

                print("COMMAND")
                print("         CL")
                print("DESCRIPTION")
                print("         Current location of the robot arm")

                print("COMMAND")
                print("         RS")
                print("DESCRIPTION")
                print("         Move the robot arm and moving bar to their initial position and rotation")
                
                print("COMMAND")
                print("         HR")
                print("SYNOPSIS")
                print("         HR [ angle ]")
                print("DESCRIPTION")
                print("         Controller rotates horizontally by angle")
                
                print("COMMAND")
                print("         FC")
                print("SYNOPSIS")
                print("         FC [ M (moving bar) ] [ {F(front), B(back), L(left), R(right)} ] [ distance ]")
                print("         FC [ R (robot arm) ] [ {R(rall), P(pitch), Y(yaw)} ] [ angle ]")
                print("         FC [ R (robot arm) ] [ {x(x_axis), y(y_axis), z(z_axis)} ] [ distance ]")
                print("DESCRIPTION")    
                print("         Move and rotate the robot arm and moving bar")         


                print("-------------------------------------------------------------------------------------------")
                continue

            elif input_cmd == "":
                print("Not entered.")
                continue
            
            trans_cmd_func(client_socket, cmd)

            data = client_socket.recv(1024)

            print("Server >> ", repr(data.decode('utf-8')))
            print(type(data))

        except KeyboardInterrupt:
            print('Disconnect Server ')
            client_socket.close()
            sys.exit()
            
# test code)
# send_cmd_to_server("CL")
# send_cmd_to_server("HR 20")
# send_cmd_to_server("RS")
# send_cmd_to_server("CL")
# send_cmd_to_server("FC x 10 y -10 Y 10")
# send_cmd_to_server("CL")
# send_cmd_to_server("CLC")
# send_cmd_to_server("FC xx 22")
# send_cmd_to_server("HR xx 22")