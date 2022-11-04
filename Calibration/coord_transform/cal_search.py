from cal_main import *
from cal_def import *
import math

def cal_diff_data():
    length = len(leds_dic['pts'])
    pts_final = []

    for i in range(length):

        p_o = vector3(leds_dic['pts_origin_offset'][i]['pos'][0], leds_dic['pts_origin_offset'][i]['pos'][1], leds_dic['pts_origin_offset'][i]['pos'][2])
        p_n = vector3(leds_dic['pts_noise_refactor'][i]['hcoord'][0], leds_dic['pts_noise_refactor'][i]['hcoord'][1], leds_dic['pts_noise_refactor'][i]['hcoord'][2])

        tmp_p_o = transfer_point_inverse(p_o, offset)
        tmp_p_n = transfer_point_inverse(p_n, offset)

        x = tmp_p_o.x - tmp_p_n.x
        y = tmp_p_o.y - tmp_p_n.y
        z = tmp_p_o.z - tmp_p_n.z

        '''        
        x = leds_dic['pts_origin_offset'][i]['pos'][0] - leds_dic['pts_noise_refactor'][i]['hcoord'][0]
        y = leds_dic['pts_origin_offset'][i]['pos'][1] - leds_dic['pts_noise_refactor'][i]['hcoord'][1]
        z = leds_dic['pts_origin_offset'][i]['pos'][2] - leds_dic['pts_noise_refactor'][i]['hcoord'][2]
        '''

        rx = leds_dic['pts_noise'][i]['pos'][0] + x
        ry = leds_dic['pts_noise'][i]['pos'][1] + y
        rz = leds_dic['pts_noise'][i]['pos'][2] + z

        u = leds_dic['pts_noise'][i]['dir'][0]
        v = leds_dic['pts_noise'][i]['dir'][1]
        w = leds_dic['pts_noise'][i]['dir'][2]

        diff_x = leds_dic['pts'][i]['pos'][0] - leds_dic['pts_noise'][i]['pos'][0]
        diff_y = leds_dic['pts'][i]['pos'][1] - leds_dic['pts_noise'][i]['pos'][1]
        diff_z = leds_dic['pts'][i]['pos'][2] - leds_dic['pts_noise'][i]['pos'][2]

        if debug == ENABLE:
            print('1.[', i, ']: x(', diff_x, ') y(', diff_y, ') z(', diff_z, ')')
            print('2.[', i, ']: x(', x, ') y(', y, ') z(', z, ')')

        pts_final.append({'idx': i, 'pos': list(map(float, [rx, ry, rz])),
                          'dir': list(map(float, [u, v, w])),
                          'pattern': leds_dic['pts'][i]['pattern']})

        leds_dic['pts_final'] = pts_final

# remake offset
def set_offset(offset):
    global offset_refactor
    offset_refactor = offset

def cal_data():
    plt.style.use('dark_background')
    fig_cam_facing = plt.figure(figsize=(10, 10))

    # 시뮬레이션을 위한 곳이다.
    for idx, leds in enumerate(trans_leds_array):
        fig_cam_facing.tight_layout()
        ax_cf = fig_cam_facing.add_subplot(4, round(cam_len / 4), idx + 1)
        ax_cf.set_title(f'origin cam {idx}')

        origin_pts = []
        origin_leds = [[x['pos'][0], x['pos'][1], x['pos'][2]] for x in leds['pts_facing']]

        for lpt in origin_leds:
            pt = vector3(lpt[0], lpt[1], lpt[2])
            # ToDo find remake offset
            temp_pt = transfer_point(pt, offset)
            # temp_pt = transfer_point(pt, offset_refactor)
            origin_pts.append([temp_pt.x, temp_pt.y, temp_pt.z])

        x, y = cam_projection(idx, np.array(origin_pts), camera_array[idx], ax_cf, DISABLE, 'yellow')

        noise_pts = []
        noise_leds = [[x['pos'][0], x['pos'][1], x['pos'][2]] for x in leds['pts_noise_facing']]
        for lpt in noise_leds:
            pt = vector3(lpt[0], lpt[1], lpt[2])
            temp_pt = transfer_point(pt, offset)
            noise_pts.append([temp_pt.x, temp_pt.y, temp_pt.z])

        nx, ny = cam_projection(idx, np.array(noise_pts), camera_array[idx], ax_cf, ENABLE, 'red')

        if len(x) != len(nx):
            print('project count not equal: ', len(x), ' vs ', len(nx))

        # led number가 같지 않으면 의미 없음
        led_num = np.array([x['idx'] for x in leds['pts_facing']])
        led_2d = [{'idx': led_num, 'x': x, 'y': y, 'nx': nx, 'ny': ny}]

        algo_coord_three(led_2d, idx)
     #end for
    cal_diff_data()

def make_noise_data():
    pts_origin = []
    pts_noise = []

    origin_data = leds_dic['pts']
    print('offset_refactor: ', offset_refactor)
    for idx, data in enumerate(origin_data):
        pt = vector3(data['pos'][0], data['pos'][1], data['pos'][2])
        # ToDo find remake offset
        temp_pt = transfer_point(pt, offset)
        # temp_pt = transfer_point(pt, offset_refactor)
        pts_origin.append({'idx': idx, 'pos': list(map(float, [temp_pt.x, temp_pt.y, temp_pt.z])),
                           'dir': list(map(float, [data['dir'][0], data['dir'][1], data['dir'][2]])),
                           'pattern': data['pattern']})
    leds_dic['pts_origin_offset'] = pts_origin

    noise_data = leds_dic['pts_noise']
    for idx, data in enumerate(noise_data):
        pt = vector3(data['pos'][0], data['pos'][1], data['pos'][2])
        temp_pt = transfer_point(pt, offset)
        pts_noise.append({'idx': idx, 'pos': list(map(float, [temp_pt.x, temp_pt.y, temp_pt.z])),
                          'dir': list(map(float, [data['dir'][0], data['dir'][1], data['dir'][2]])),
                          'pattern': data['pattern']})
    leds_dic['pts_noise_offset'] = pts_noise

def change_coord(coord, cam_pose):
    pt = copy.deepcopy(cam_pose)
    pt['position'] = vector3(0, 0, 0)
    new_pt = transfer_point_inverse(coord, pt)
    return new_pt

def algo_coord_three(origin, cam_num):
    for idx, leds in enumerate(origin):
        length = len(leds['idx'])
        set_dis = 0.5
        for i in range(length):
            led_num = leds['idx'][i]
            print('led num: ', led_num)
            origin_coord = np.array(change_coord(vector3(leds['x'][i], leds['y'][i], set_dis), camera_array[cam_num]))
            noise_coord = np.array(change_coord(vector3(leds['nx'][i], leds['ny'][i], set_dis), camera_array[cam_num]))

            if debug == ENABLE:
                print('origin')
                print(camera_array[cam_num], '\n',
                      '<', round(leds['x'][i], 8), ',', round(leds['y'][i], 8), '>\n',
                      '<', origin_coord[0], ',', origin_coord[1], ',', origin_coord[2], '>\n',

                      '[', leds_dic['pts_origin_offset'][led_num]['pos'][0], ',',
                      leds_dic['pts_origin_offset'][led_num]['pos'][1], ',',
                      leds_dic['pts_origin_offset'][led_num]['pos'][2], ']')

                print('noise')
                print(camera_array[cam_num], '\n',
                      '<', round(leds['nx'][i], 8), ',', round(leds['ny'][i], 8), '>\n',
                      '<', noise_coord[0], ',', noise_coord[1], ',', noise_coord[2], '>\n',
                      '[', leds_dic['pts_noise_offset'][led_num]['pos'][0], ',',
                      leds_dic['pts_noise_offset'][led_num]['pos'][1], ',',
                      leds_dic['pts_noise_offset'][led_num]['pos'][2], ']')

            pre_cam_status = leds_dic['pts_noise_refactor'][led_num]['cam']
            if pre_cam_status != -1 and pre_cam_status < len(camera_array):
                print('exist pre coords')
                hcoord = leds_dic['pts_noise_refactor'][led_num]['hcoord']
                pre_cam_num = leds_dic['pts_noise_refactor'][led_num]['cam']

                # cam postion 계산
                pre_cam_pt = camtoworld_(camera_array[pre_cam_num]['position'], camera_array[pre_cam_num])
                cur_cam_pt = camtoworld_(camera_array[cam_num]['position'], camera_array[cam_num])
                if debug == ENABLE:
                    print(hcoord, ' cam: ', pre_cam_num)
                    print('pre_cam:', pre_cam_pt, 'cur_cam:', cur_cam_pt)
                # Z는 나중에 구함
                # pre cam
                x11 = hcoord[0]
                y11 = hcoord[1]
                x12 = pre_cam_pt.x
                y12 = pre_cam_pt.y
                # cur cam
                x21 = noise_coord[0]
                y21 = noise_coord[1]
                x22 = cur_cam_pt.x
                y22 = cur_cam_pt.y
                calc_data = 0

                cx, cy = get_crosspt(x11, y11, x12, y12, x21, y21, x22, y22)
                # 더할 필요가 없다.
                leds_dic['pts_noise_refactor'][led_num]['cam'] = len(camera_array) + 1
                x_d = noise_coord[0] - cur_cam_pt.x
                y_d = noise_coord[1] - cur_cam_pt.y
                z_d = noise_coord[2] - cur_cam_pt.z

                cz = round(math.sqrt(math.pow(cx - cur_cam_pt.x, 2) + math.pow(cy - cur_cam_pt.y, 2)) * (
                            noise_coord[2] / math.sqrt(math.pow(x_d, 2) + math.pow(y_d, 2))), 8)

                leds_dic['pts_noise_refactor'][led_num]['hcoord'][0] = cx
                leds_dic['pts_noise_refactor'][led_num]['hcoord'][1] = cy
                leds_dic['pts_noise_refactor'][led_num]['hcoord'][2] = cz
                print('cx:', cx, ' cy:', cy, ' cz:', cz)
            elif pre_cam_status == -1:
                print('pre coords not found')
                leds_dic['pts_noise_refactor'][led_num]['cam'] = cam_num
                leds_dic['pts_noise_refactor'][led_num]['hcoord'][0] = noise_coord[0]
                leds_dic['pts_noise_refactor'][led_num]['hcoord'][1] = noise_coord[1]
                leds_dic['pts_noise_refactor'][led_num]['hcoord'][2] = noise_coord[2]
            else:
                print('skip')

            if debug == ENABLE:
                print(leds_dic['pts_noise_refactor'][led_num])
                print('return')
    # cross_test()

def draw_result():
    plt.style.use('default')
    fig = plt.figure(figsize=(10, 10))
    ax_new = fig.add_subplot(111, projection='3d')

    if debug == ENABLE:
        for idx, leds in enumerate(leds_dic['pts_noise_offset']):
            print(idx, '[o]:', leds['pos'][0], ' ', leds['pos'][1], ' ', leds['pos'][2])
            print(idx, '[1]:', leds_dic['pts_noise_refactor'][idx]['hcoord'][0], ' ',
                  leds_dic['pts_noise_refactor'][idx]['hcoord'][1], ' ', leds_dic['pts_noise_refactor'][idx]['hcoord'][2])

    #draw_dots(3, leds_dic['pts_origin_offset'], ax_new, 'green')
    #draw_dots(3, leds_dic['pts_noise_refactor'], ax_new, 'black')
    draw_dots(3, leds_dic['pts_final'], ax_new, 'red')
    draw_dots(3, leds_dic['pts_noise'], ax_new, 'gray')
    draw_dots(3, leds_dic['pts'], ax_new, 'blue')

    ax_new.set_xlim([-0.1, 0.1])
    ax_new.set_xlabel('X')
    ax_new.set_ylim([-0.1, 0.1])
    ax_new.set_ylabel('Y')
    ax_new.set_zlim([-0.1, 0.1])
    ax_new.set_zlabel('Z')

    plt.style.use('dark_background')
    fig_final = plt.figure(figsize=(10, 10))
    for idx, leds in enumerate(trans_leds_array):
        fig_final.tight_layout()
        ax_final = fig_final.add_subplot(4, round(cam_len / 4), idx + 1)
        ax_final.set_title(f'result cam {idx}')
        nidx = [x['idx'] for x in leds['pts_noise_facing']]
        if debug == ENABLE:
            print('cam num:', idx, ' ', nidx)
        origin_pts = []
        noise_pts = []
        final_pts = []
        for i in range(len(nidx)):
            origin_pts.append(
                [leds_dic['pts'][i]['pos'][0], leds_dic['pts'][i]['pos'][1], leds_dic['pts'][i]['pos'][2]])
            noise_pts.append(
                [leds_dic['pts_noise'][i]['pos'][0], leds_dic['pts_noise'][i]['pos'][1],
                 leds_dic['pts_noise'][i]['pos'][2]])
            final_pts.append(
                [leds_dic['pts_final'][i]['pos'][0], leds_dic['pts_final'][i]['pos'][1],
                 leds_dic['pts_final'][i]['pos'][2]])

        cam_projection(idx, np.array(origin_pts), camera_array[idx], ax_final, DISABLE, 'white')
        cam_projection(idx, np.array(noise_pts), camera_array[idx], ax_final, DISABLE, 'yellow')
        cam_projection(idx, np.array(final_pts), camera_array[idx], ax_final, DISABLE, 'red')

    return

def get_crosspt(x11, y11, x12, y12, x21, y21, x22, y22):
    if x12 == x11 or x22 == x21:
        print('delta x=0')
        if x12 == x11:
            cx = x12
            m2 = (y22 - y21) / (x22 - x21)
            cy = m2 * (cx - x21) + y21
            return cx, cy
        if x22 == x21:
            cx = x22
            m1 = (y12 - y11) / (x12 - x11)
            cy = m1 * (cx - x11) + y11
            return cx, cy

    m1 = (y12 - y11) / (x12 - x11)
    m2 = (y22 - y21) / (x22 - x21)
    if m1 == m2:
        print('parallel')
        return None
    print(x11, y11, x12, y12, x21, y21, x22, y22, m1, m2)
    cx = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
    cy = m1 * (cx - x11) + y11

    return round(cx, 8), round(cy, 8)

def cross_test(x11, y11, x12, y12, x21, y21, x22, y22):
    plt.figure()
    plt.plot([x11, x12], [y11, y12], c='r')
    plt.plot([x21, x22], [y21, y22], c='b')

    cx, cy = get_crosspt(x11, y11, x12, y12, x21, y21, x22, y22)
    print('cx:', cx, ' cy:', cy)
    plt.plot(cx, cy, 'ro')
    plt.show()

def coord_cal_algo():
    print('coord_cal_algo')
    make_noise_data()
    cal_data()
    draw_result()

    #Test Code

