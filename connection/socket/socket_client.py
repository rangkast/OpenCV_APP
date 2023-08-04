from socket_def import *

# test_set.offset = {'x':23.34, 'y': 40.00, 'z':42.30, 'a':5, 'b':7.0, 'c':1}   # offset 변경
# test_set.save_itv = 100.3

# setting cmd -> offset 만 변경 시
test_set = Setting_CMD()
# #
test_set.mv_sp = 200
test_set.save_itv = 200  # meec
# test_set.offset = {'x': 0, 'y': 50, 'z': 0, 'a': 20, 'b': 20, 'c': 20}
# test_set.save_rb = 'n'
test_set.ar_coord = 'ac'
# test_set.ar_coord = 'rc'
# ar_coord = 'rc'

test_set.trans_setting()    # 나머지는 setting 값은 이전 값 그대로(default) offset만 변경
send_cmd_to_server(sys_set) # robot의 setting 값 확인
# send_cmd_to_server(cpt)

# trans_joint({'1':0, '2':0, '3':-3, '4':0, '5':0, '6':0})
# trans_rpos({'x':-30, 'y':0, 'z':0, 'a':0, 'b':0, 'c':0})
# trans_rpos({'x': 540.00, 'y': -210.00, 'z': 300.00, 'a': 160.00, 'b': 70.00, 'c': 120.00})
# trans_rpos({'x': 480.00, 'y': -210.00, 'z': 290.00, 'a': 180.00, 'b': 60.00, 'c': 120.00})
# send_cmd_to_server(cpar)
# send_cmd_to_server(sys_set)
# trans_joint({'1': -45.54, '2': 57.26, '3': 32.92, '4': 0.85, '5': -86.75, '6': 98.50})
# trans_tbpos({'x':550, 'y':0, 'z':0})
# send_cmd_to_server(cpt)
# trans_sbpos({'y':0, 'z':1.11})



# rb_info.show_all_data()