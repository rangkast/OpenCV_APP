import sys
import os
import numpy as np

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Add the directory containing poselib to the module search path
print(script_dir)
sys.path.append(os.path.join(script_dir, '../../../../EXTERNALS'))

import poselib
# help(poselib.p3p)



target_pose_led_data = np.array([
     [-0.01108217, -0.00278021, -0.01373098],
     [-0.02405356,  0.00777868, -0.0116913 ],
     [-0.02471722,  0.00820648,  0.00643996],
     [-0.03312733,  0.02861635,  0.0013793 ],
     [-0.02980387,  0.03374299,  0.02675826],
     [-0.0184596,  0.06012725,  0.02233215],
     [-0.00094422,  0.06020401,  0.04113377],
     [ 0.02112556,  0.06993855,  0.0256014 ],
     [ 0.04377158,  0.05148328,  0.03189337],
     [ 0.04753083,  0.05121397,  0.01196245],
     [ 0.0533449,  0.02829823, 0.01349697],
     [ 0.05101214,  0.02247323, -0.00647229],
     [ 0.04192879,  0.00376628, -0.00139432],
     [ 0.03947314,  0.00479058, -0.01699771],
     [ 0.02783124, -0.00088511, -0.01754906],
])

image_coordinates = np.array([
    [657.4802894356005, 504.83178002894357],
    [631.7905127712694, 500.90293835221814],
    [630.633866114346, 459.46895150259877],
    # [581.3196922716902, 470.61686070321184]
])

# 우선 입력 데이터를 numpy.array로 변환합니다.
target_pose_led_data = np.array(target_pose_led_data)
image_coordinates = np.array(image_coordinates)

# 2D 포인트를 정규화하고, 각각에 z=1을 추가합니다.
# 이를 위해 numpy.hstack을 사용하여 2D 좌표와 1로 이루어진 열을 합칠 수 있습니다.
x = np.hstack((image_coordinates, np.ones((image_coordinates.shape[0], 1))))

# 3D 포인트는 이미 적절한 형태로 제공되므로 변환할 필요가 없습니다.
X = target_pose_led_data

# 이제 x와 X를 p3p 메서드에 넣을 수 있습니다.
result = poselib.p3p(x, X)

print(result)