cmake_minimum_required(VERSION 3.18.1)

project("fastcv_app")

set(pathPROJECT D:/Files/app/fastcv_app)

set(pathExternalLibs ${pathPROJECT}/app/src/main/libs)
set(pathFastCV ${pathExternalLibs}/fastcv)
set(pathFastCVLib ${pathFastCV}/${ANDROID_ABI}/libfastcv.a)

set(pathOPENCV ${pathPROJECT}/sdk)
set(pathLIBOPENCV_JAVA ${pathOPENCV}/native/libs/${ANDROID_ABI}/libopencv_java4.so)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")

add_library(fastcv_app SHARED
            native-lib.cpp
            FastCVSample.cpp
            FPSCounter.cpp
            CameraRendererRGB565GL2.cpp
            CameraUtil.cpp
            FastCVSampleRenderer.cpp

        )

include_directories(${pathFastCV}/inc)
add_library(lib_fastcv SHARED IMPORTED )
set_target_properties(lib_fastcv PROPERTIES IMPORTED_LOCATION ${pathFastCVLib})

include_directories(${pathOPENCV}/native/jni/include)
add_library(lib_opencv SHARED IMPORTED )
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${pathLIBOPENCV_JAVA})

find_library(log-lib log)

target_link_libraries( # Specifies the target library.
        fastcv_app
        lib_opencv
        lib_fastcv
        GLESv2
        ${log-lib})