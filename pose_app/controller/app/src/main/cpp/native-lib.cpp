#include <jni.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <android/log.h>

#include "detect_obj.h"
#include "point_tracker.h"
#include "point_obj.h"

using namespace cv;

//https://sungcheol-kim.gitbook.io/jni-tutorial/chapter18
//jni reference
extern std::vector<cv::Point3d> objectPoints;

Mat gR;
Mat gt;
Mat gr;

point_obj objLeft;
pt_camera_info info;

extern "C"
JNIEXPORT void JNICALL
Java_com_opencv_controller_MainActivity_InitMatrix(JNIEnv *env, jobject thiz) {
    // TODO: implement InitMatrix()
    gR = Mat::eye(3, 3, CV_64FC1);
    gt = Mat::zeros(3, 1, CV_64FC1);
    gr = Mat::zeros(3, 1, CV_64FC1);
    objLeft.init();
}

extern "C"
JNIEXPORT void JNICALL
Java_com_opencv_controller_MainActivity_GetRvecTvec(JNIEnv *env, jobject thiz,
                    jfloatArray mat_addr_rvec, jfloatArray mat_addr_tvec){

    float fRvec[3] = {};
    float fTvec[3] = {};

    fRvec[0] = (float)gr.at<double>(0, 0);
    fRvec[1] = (float)gr.at<double>(0, 1);
    fRvec[2] = (float)gr.at<double>(0, 2);
    fTvec[0] = (float)gt.at<double>(0, 0);
    fTvec[1] = (float)gt.at<double>(0, 1);
    fTvec[2] = (float)gt.at<double>(0, 2);
    (*env).SetFloatArrayRegion(mat_addr_rvec, 0, 3, fRvec);
    (*env).SetFloatArrayRegion(mat_addr_tvec, 0, 3, fTvec);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_opencv_controller_MainActivity_NativeInfo(JNIEnv *env, jobject thiz, jfloatArray mat_addr_info) {
    float valuesf[16] = {};

    for (int i = 0; i <= 8; i++) {
        valuesf[i] = info.rv[i];
    }
    for (int i = 9; i <= 11; i++) {
        valuesf[i] = info.tv[i - 9];
    }
    valuesf[12] = info.contours;
    valuesf[13] = info.start_tracking;
    valuesf[14] = info.solutions;

    (*env).SetFloatArrayRegion(mat_addr_info, 0, 16, valuesf);
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_opencv_controller_MainActivity_ConvertRGBtoGray(JNIEnv *env, jobject thiz,
                            jlong mat_addr_input, jlong mat_addr_result
                            ,jfloatArray mat_addr_rotation, jintArray cInfo
                            ) {
    // TODO: implement ConvertRGBtoGray()
    Mat &matInput = *(Mat *)mat_addr_input;
    Mat &matResult = *(Mat *)mat_addr_result;

    bool bResL = false;
    matResult = matInput;

    point_tracker pt;

    // Set Camera Info
    jint *cfArray = env->GetIntArrayElements(cInfo, nullptr);
//    __android_log_print(ANDROID_LOG_DEBUG, APPNAME,"cInfo %d %d %d %d", cfArray[0], cfArray[1], cfArray[2], cfArray[3]);


    info.R = Mat::eye(3, 3, CV_64FC1);
    info.rvec = Mat::zeros(3, 1, CV_64FC1);
    info.tvec = Mat::zeros(3, 1, CV_64FC1);

    info.rv = info.R.ptr<double>();
    info.tv = info.tvec.ptr<double>();

    info.fov = (float)cfArray[2];
    info.res_x = cfArray[1];
    info.res_y = cfArray[0];
    info.touch_status = cfArray[3];
    // Get Controller points
    std::vector <Point2f> points = {};
    detect_obj dobj;

    dobj.detect_points(info, matInput, matResult, points);

    if (points.size() != 0) {
        objLeft.clearPoints();
        for (int i = 0; i < points.size(); i++) {
            circle(matResult, points[i], 5, Scalar(255, 255, 255), 10, 8, 0);
            objLeft.push_back(points[i]);
        }
     //     bResL = pt.track(objLeft.points, info, &objLeft.gR, &objLeft.gr, &objLeft.gt);
        bResL = pt.estimate_pose(info, matResult, objLeft.points, &objLeft.gR, &objLeft.gr, &objLeft.gt);
    } else {
        __android_log_print(ANDROID_LOG_DEBUG, APPNAME,
                            "points size Error 0: Pass pt.track");
    }

    float valuesf[16] = {};

    double* l0 = objLeft.gR.ptr<double>(0);
    double* l1 = objLeft.gR.ptr<double>(1);
    double* l2 = objLeft.gR.ptr<double>(2);
    double* t1 = objLeft.gt.ptr<double>(0);

    valuesf[0] = (float)l0[0];
    valuesf[1] = (float)l0[1];
    valuesf[2] = (float)l0[2];
    valuesf[3] = 0;

    valuesf[4] = (float)l1[0];
    valuesf[5] = (float)l1[1];
    valuesf[6] = (float)l1[2];
    valuesf[7] = 0;

    valuesf[8] = (float)l2[0];
    valuesf[9] = (float)l2[1];
    valuesf[10] = (float)l2[2];
    valuesf[11] = 0;

    // On MyGLRenderer camera eyeZ is moved.so translation is multiplied by the ratio
    valuesf[12] = (float)t1[0];
    valuesf[13] = (float)t1[1];
    valuesf[14] = (float)t1[2];
    valuesf[15] = 1.f;

    (*env).SetFloatArrayRegion(mat_addr_rotation, 0, 16, valuesf);

    /*
    __android_log_print(ANDROID_LOG_DEBUG, APPNAME,
                        "\nvaluesf, %f, %f, %f, %f\nvaluesf, %f, %f, %f, %f\nvaluesf, %f, %f, %f, %f\nvaluesf, %f, %f, %f, %f\n",
                        valuesf[0], valuesf[1], valuesf[2], valuesf[3],
                        valuesf[ 4], valuesf[5], valuesf[6], valuesf[7],
                        valuesf[ 8], valuesf[9], valuesf[10], valuesf[11],
                        valuesf[12], valuesf[13], valuesf[14], valuesf[15]
    );
    */


    return bResL;
}

