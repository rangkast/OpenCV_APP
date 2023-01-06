#include <jni.h>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <fastcv.h>
#include <android/log.h>

#define LOG_TAG    "native-lib.cpp"
#define DPRINTF(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)

extern "C"
JNIEXPORT jstring JNICALL
Java_project_test_fastcv_1app_SplashScreen_getFastCVVersion(JNIEnv *env, jobject thiz) {
    // TODO: implement getFastCVVersion()
    char sVersion[32];
    fcvGetVersion(sVersion, 32);
    DPRINTF("FastCV version %s", sVersion);
    return env->NewStringUTF(sVersion);
}