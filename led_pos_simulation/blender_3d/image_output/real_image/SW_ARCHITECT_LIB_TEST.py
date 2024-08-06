# .pyd library test
import SW_ARCHITECT_lib

SW_ARCHITECT_lib.BA_RT(info_name='CAMERA_INFO.pickle', save_to='BA_RT.pickle', target='BLENDER')



'''

1. Microsoft c++ BuildTool 필요
    - cl.exe 동작과 path를 확인해 볼 것

2. Windows에서는 Python 확장 모듈이 .pyd 파일로 생성
이는 .so 파일과 동일한 기능 
.pyd 파일은 Windows에서 동적으로 로드 가능한 모듈로 사용됨

'''