from Advanced_Function import *
from pymongo import MongoClient
import numpy as np

def convert_ndarray_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {k: convert_ndarray_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(i) for i in data]
    else:
        return data

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

    # numpy.ndarray를 리스트로 변환
    gathering_data_to_store = {
        "origin_data": convert_ndarray_to_list(origin_data),
        "model_data": convert_ndarray_to_list(model_data),
        "direction": convert_ndarray_to_list(direction),
        "camera_info_backup": convert_ndarray_to_list(camera_info_backup),
        "new_camera_info_up": convert_ndarray_to_list(new_camera_info_up),
        "new_camera_info_up_keys": convert_ndarray_to_list(new_camera_info_up_keys),
        "new_camera_info_down": convert_ndarray_to_list(new_camera_info_down),
        "new_camera_info_down_keys": convert_ndarray_to_list(new_camera_info_down_keys),
    }

    camera_info_data_to_store = {
        "camera_info": convert_ndarray_to_list(CAMERA_INFO)
    }

    transform_data_to_store = {
        "rigid_3d_transform_iqr": convert_ndarray_to_list(RIGID_3D_TRANSFORM_IQR)
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



몽고DB와 파이썬
최종 업데이트 날짜: 2022년 12월 28일
전제조건 : MongoDB : 소개
MongoDB는 컬렉션과 문서의 개념을 기반으로 작동하는 크로스 플랫폼 문서 지향 데이터베이스입니다. MongoDB는 고속, 고가용성, 높은 확장성을 제공합니다.
사람들의 마음 속에 떠오르는 다음 질문은 “왜 MongoDB인가?”입니다.
MongoDB를 선택하는 이유:

계층적 데이터 구조를 지원합니다. ( 자세한 내용은 문서를 참조하세요.)
Python의 사전과 같은 연관 배열을 지원합니다.
Python 애플리케이션을 데이터베이스와 연결하기 위한 내장 Python 드라이버입니다. 예 - PyMongo
빅데이터용으로 설계되었습니다.
MongoDB 배포는 매우 쉽습니다.







MongoDB를 이용한 데이터 저장 전략은 다양한 소프트웨어 아키텍처 이점을 제공합니다. 이를 통해 시스템의 성능, 확장성, 유연성, 데이터 관리 용이성 등이 향상됩니다. 아래는 MongoDB를 사용함으로써 얻을 수 있는 주요 이점과 아키텍처 전략입니다.

### 주요 이점

1. **스키마리스 데이터 모델**
   - **유연성**: MongoDB는 스키마리스 데이터베이스로, 데이터 구조를 사전에 정의할 필요가 없습니다. 이는 애플리케이션 요구사항이 변경될 때 데이터베이스 스키마를 수정할 필요 없이 데이터 모델을 유연하게 변경할 수 있습니다.
   - **빠른 개발**: 데이터 구조가 자주 변경되거나 진화하는 애플리케이션 개발에 이상적입니다.

2. **확장성**
   - **수평적 확장**: MongoDB는 샤딩을 통해 수평적으로 확장할 수 있습니다. 이는 데이터베이스 성능을 유지하면서 대규모 데이터를 처리하고 저장할 수 있습니다.
   - **성능 향상**: 분산된 데이터 저장을 통해 데이터 접근 속도를 높일 수 있습니다.

3. **고가용성**
   - **복제**: MongoDB는 복제를 통해 데이터의 가용성과 안정성을 높입니다. 다중 복제본을 유지함으로써 장애 발생 시 자동으로 장애 조치가 가능합니다.
   - **데이터 손실 방지**: 여러 서버에 데이터를 복제함으로써 데이터 손실 가능성을 줄입니다.

4. **강력한 쿼리 언어**
   - **유연한 쿼리**: MongoDB는 강력한 쿼리 언어를 제공하여 데이터를 쉽게 검색, 필터링, 정렬할 수 있습니다.
   - **Aggregation Framework**: 복잡한 데이터 처리 및 분석 작업을 수행할 수 있는 도구를 제공합니다.

5. **JSON 기반의 직관적인 데이터 모델**
   - **사용 용이성**: JSON 포맷을 사용하여 데이터 구조가 직관적이며 이해하기 쉽습니다.
   - **다양한 데이터 타입 지원**: 문자열, 숫자, 배열, 객체 등 다양한 데이터 타입을 지원하여 복잡한 데이터 구조를 쉽게 저장할 수 있습니다.

### 아키텍처 전략

1. **데이터 중심 아키텍처**
   - **데이터 통합**: MongoDB를 사용하면 다양한 형태의 데이터를 통합하여 관리할 수 있습니다. JSON 형식의 문서 데이터 모델을 사용하여 여러 소스에서 수집한 데이터를 통합할 수 있습니다.
   - **중앙 집중식 데이터 저장소**: 데이터 중심의 접근 방식으로 중앙에서 데이터를 관리하고 필요한 모든 서비스와 애플리케이션에서 동일한 데이터를 사용할 수 있습니다.

2. **마이크로서비스 아키텍처와의 통합**
   - **독립적인 데이터 관리**: 각 마이크로서비스가 독립적으로 MongoDB를 사용하여 자신만의 데이터를 관리할 수 있습니다. 이를 통해 서비스 간의 결합도를 낮추고 확장성을 높일 수 있습니다.
   - **서비스 장애 격리**: 하나의 서비스 장애가 다른 서비스에 영향을 미치지 않도록 데이터와 서비스를 독립적으로 관리할 수 있습니다.

3. **데이터 분석 및 실시간 처리**
   - **실시간 데이터 처리**: MongoDB의 Aggregation Framework를 사용하여 실시간 데이터 분석 및 처리가 가능합니다. 대용량 데이터에 대한 실시간 분석 요구사항을 충족할 수 있습니다.
   - **빅데이터 처리**: 대규모 데이터 분석 작업을 수행할 때 MongoDB를 활용하여 효율적으로 데이터를 저장하고 처리할 수 있습니다.

4. **데이터 보안 및 접근 제어**
   - **역할 기반 접근 제어**: MongoDB는 역할 기반 접근 제어(RBAC)를 제공하여 데이터에 대한 접근을 세밀하게 제어할 수 있습니다.
   - **암호화**: 데이터 암호화를 통해 저장된 데이터의 보안을 강화할 수 있습니다.

### 결론

MongoDB를 데이터 저장소로 사용하는 것은 다양한 이점을 제공하며, 특히 유연성과 확장성이 중요한 애플리케이션에서 큰 효과를 발휘합니다. 스키마리스 데이터 모델, 수평적 확장성, 고가용성, 강력한 쿼리 언어 등의 기능을 통해 개발자와 데이터베이스 관리자가 더 효율적이고 생산적으로 작업할 수 있습니다. 이러한 이점들은 데이터 중심의 소프트웨어 아키텍처에서 특히 유용하며, 마이크로서비스 아키텍처와의 통합에서도 많은 장점을 제공합니다.






'''