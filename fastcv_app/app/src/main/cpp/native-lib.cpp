#include <jni.h>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <fastcv.h>
#include <android/log.h>

#define LOG_TAG    "native-lib.cpp"
#define DPRINTF(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)
extern "C" JNIEXPORT jstring JNICALL
Java_project_test_fastcv_1app_MainActivity_stringFromJNI
        (
        JNIEnv* env,
        jobject /* this */
        ) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C"  JNIEXPORT jstring JNICALL
Java_project_test_fastcv_1app_MainActivity_getFastCVVersion
        (
                JNIEnv* env,
                jobject obj
        ) {
    char sVersion[32];
    fcvGetVersion(sVersion, 32);
    DPRINTF("FastCV version %s", sVersion);
    return env->NewStringUTF(sVersion);
}

