from Advanced_Function import *
from pymongo import MongoClient

def main():
    # MongoDB에 연결
    client = MongoClient('localhost', 27017)
    db = client['calibration_db']  # 사용할 데이터베이스 이름
    gathering_collection = db['gathering_data']  # 데이터 수집 관련 컬렉션
    camera_info_collection = db['camera_info']  # 카메라 정보 관련 컬렉션
    transform_collection = db['transform_data']  # 변환 데이터 관련 컬렉션

    origin_data, direction = init_coord_json(f"{script_dir}/jsons/specs/arcturas_right.json")
    model_data, direction = init_coord_json(f"{script_dir}/jsons/specs/arcturus_#3_right+.json")

    # pickle 데이터 읽기
    camera_info_backup = pickle_data(READ, "CAMERA_INFO_PLANE.pickle", None)['CAMERA_INFO']
    new_camera_info_up = pickle_data(READ, "NEW_CAMERA_INFO_1.pickle", None)['NEW_CAMERA_INFO']
    new_camera_info_up_keys = list(new_camera_info_up.keys())
    new_camera_info_down = pickle_data(READ, "NEW_CAMERA_INFO_-1.pickle", None)['NEW_CAMERA_INFO']
    new_camera_info_down_keys = list(new_camera_info_down.keys())
    
    CAMERA_INFO = pickle_data(READ, 'CAMERA_INFO.pickle', None)['CAMERA_INFO.pickle'.split('.')[0]]
    RIGID_3D_TRANSFORM_IQR = pickle_data(READ, 'RIGID_3D_TRANSFORM.pickle', None)['IQR_ARRAY_LSM']

    # MongoDB에 데이터 저장
    gathering_data_to_store = {
        "origin_data": origin_data,
        "model_data": model_data,
        "direction": direction,
        "camera_info_backup": camera_info_backup,
        "new_camera_info_up": new_camera_info_up,
        "new_camera_info_up_keys": new_camera_info_up_keys,
        "new_camera_info_down": new_camera_info_down,
        "new_camera_info_down_keys": new_camera_info_down_keys,
    }

    camera_info_data_to_store = {
        "camera_info": CAMERA_INFO
    }

    transform_data_to_store = {
        "rigid_3d_transform_iqr": RIGID_3D_TRANSFORM_IQR
    }

    gathering_collection.insert_one(gathering_data_to_store)
    camera_info_collection.insert_one(camera_info_data_to_store)
    transform_collection.insert_one(transform_data_to_store)

    print("Data has been stored in MongoDB")

if __name__ == "__main__":
    main()
    
    
'''
예, `CAMERA_INFO`와 `RIGID_3D_TRANSFORM_IQR`를 다른 컬렉션에 저장할 수 있습니다. MongoDB에서는 각 컬렉션이 독립적으로 존재할 수 있으며, 데이터를 별도로 저장하고 관리할 수 있습니다. 이를 통해 데이터를 논리적으로 분리하고, 필요할 때 쉽게 접근할 수 있습니다.


### 데이터베이스 구조

다음은 MongoDB 데이터베이스 구조를 시각화한 것입니다:

```plaintext
calibration_db
├── gathering_data
│   ├── document
│       ├── origin_data
│       ├── model_data
│       ├── direction
│       ├── camera_info_backup
│       ├── new_camera_info_up
│       ├── new_camera_info_up_keys
│       ├── new_camera_info_down
│       ├── new_camera_info_down_keys
│
├── camera_info
│   ├── document
│       ├── camera_info
│
└── transform_data
    ├── document
        ├── rigid_3d_transform_iqr
```

### 그림으로 표현

```
+---------------------+
|  calibration_db     |
| +-----------------+ |
| | gathering_data  | |
| |  +-------------+ | 
| |  | document    | |
| |  | - origin_data         | |
| |  | - model_data          | |
| |  | - direction           | |
| |  | - camera_info_backup  | |
| |  | - new_camera_info_up  | |
| |  | - new_camera_info_up_keys | |
| |  | - new_camera_info_down    | |
| |  | - new_camera_info_down_keys | |
| |  +-------------+ |
| +-----------------+ |
| +-----------------+ |
| | camera_info     | |
| |  +-------------+ | 
| |  | document    | |
| |  | - camera_info  | |
| |  +-------------+ |
| +-----------------+ |
| +-----------------+ |
| | transform_data  | |
| |  +-------------+ | 
| |  | document    | |
| |  | - rigid_3d_transform_iqr | |
| |  +-------------+ |
| +-----------------+ |
+---------------------+
```

이 구조에서는 데이터가 각각의 컬렉션에 독립적으로 저장되므로 관리와 접근이 용이합니다. `gathering_data` 컬렉션에는 주로 수집된 데이터와 관련된 정보가, `camera_info` 컬렉션에는 카메라 정보가, `transform_data` 컬렉션에는 3D 변환 데이터가 저장됩니다.

이러한 아키텍처를 통해 각 데이터 유형을 명확하게 분리하고, 필요할 때 쉽게 접근할 수 있습니다. 또한, 각 컬렉션은 서로 독립적이기 때문에 데이터 관리가 용이합니다.

'''