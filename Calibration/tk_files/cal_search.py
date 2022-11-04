from cal_def import *


def refactor_coord(led_num, div, origin, noise):
    


    return



def print_test_coord(ax, origin, cam_num):

    #print(trans_leds_array[cam_num]['pts_facing'])

    for idx, leds in enumerate(origin):
        length = len(leds['idx'])
        wxyz = [-1, -1, -1]
        for i in range(length):            
            if leds['idx'][i] == 20:
                #ToDo
                print('origin')
                origin_coord =  np.round(np.array(camtoworld(vector3(leds['x'][i], leds['y'][i], 1), camera_array[cam_num])), 10)
                noise_coord = np.round(np.array(camtoworld(vector3(leds['nx'][i], leds['ny'][i], 1), camera_array[cam_num])), 10)
                print(camera_array[cam_num],'\n',
                        '<',leds['x'][i], ',', leds['y'][i], '>\n',
                        '<',origin_coord[0], ',',origin_coord[1], ',', origin_coord[2], '>\n', 
                        '[',trans_leds_array[cam_num]['pts_facing'][i]['pos'][0], ',',
                            trans_leds_array[cam_num]['pts_facing'][i]['pos'][1], ',',
                            trans_leds_array[cam_num]['pts_facing'][i]['pos'][2], ']')         
                
                print('noise')                
                print(camera_array[cam_num],'\n',
                        '<',leds['nx'][i], ',', leds['ny'][i], '>\n',
                        '<',noise_coord[0], ',',noise_coord[1], ',', noise_coord[2], '>\n', 
                        '[',trans_leds_array[cam_num]['pts_noise_facing'][i]['pos'][0], ',',
                            trans_leds_array[cam_num]['pts_noise_facing'][i]['pos'][1], ',',
                            trans_leds_array[cam_num]['pts_noise_facing'][i]['pos'][2], ']')

                # xyz
                div = 0
                direction = 0
                for index in range(3):
                    if abs(origin_coord[index]) != 1.0:
                        div_index = origin_coord[index] / trans_leds_array[cam_num]['pts_facing'][i]['pos'][index]                    
                        div += div_index
                    else:
                        direction = index    

                div = div/2
                print(div)

                #ToDo 값 이상함
                
                for index in range(3):
                    if index != direction:
                        wxyz[index] = origin_coord[index] / div

                print(wxyz)

                print('\n')

                
    return
#end print_test_coord()

def check_diff(ax, origin, cam_num):
    #print('origin: ', origin)
    #print('noise: ', noise)
   
    #test code
    print_test_coord(ax, origin, cam_num)

    return


