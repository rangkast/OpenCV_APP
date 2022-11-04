//
// Created by administrator on 2021. 12. 21..
//
#include <android/log.h>
#include <opencv2/opencv.hpp>

#include "detect_obj.h"

using namespace cv;

void detect_obj::detect_points(pt_camera_info info, Mat &matInput, Mat &matResult, std::vector<Point2f> &points) {
    // Detect Green Objects for TEST before IR LED Controller
    //https://sikaleo.tistory.com/84
    Mat img_hsv(matInput.rows, matInput.cols, CV_8UC4);
    cvtColor(matInput, img_hsv, COLOR_BGR2HSV);

    int hue_green = 60; // 40 when home + night
    Mat img_mask;
    inRange(img_hsv, Scalar(hue_green - 10, 100, 100), Scalar(hue_green + 10, 255, 255), img_mask);
//    int hue_green = 245;
//    Mat img_mask;
//    inRange(img_hsv, Scalar(hue_green - 10, 200, 200), Scalar(hue_green + 10, 255, 255), img_mask);

//https://cafepurple.tistory.com/64
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(img_mask, img_hsv, MORPH_DILATE, kernel, Point(0, 0), 3);
//    morphologyEx(img_mask, matResult, MORPH_DILATE, kernel, Point(0, 0), 3);

//https://fwanggu-lee.tistory.com/12
    //vector<Point> Point 객체를 요소로 갖는 벡터 어레이 선언. x,y좌표 데이터가 한 세트로 저장
    std::vector <std::vector<Point>> contours;
    //vector<vector<Point>> x,y 좌표 데이터 세트를 한 요소로, 컨투어를 표현하는 점들의 집합.
    findContours(img_mask, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
 //   __android_log_print(ANDROID_LOG_DEBUG, APPNAME, "SIZE : contours : %d", (int) contours.size());
    info.contours = contours.size();
    if (contours.size() != 0) {
//        __android_log_print(ANDROID_LOG_DEBUG, APPNAME, "contours size %d, point 1 size %d, %d, %d",
//                            (int) contours.size(), (int) contours[0].size(),
//                            (int) contours[0][0].x, (int) contours[0][0].y);

        std::vector <Moments> mu(contours.size());
        for (int i = 0; i < contours.size(); i++) {
            mu[i] = moments(contours[i]);
        }
  //      std::vector <Point2f> mc(contours.size());

        for (size_t i = 0; i < contours.size(); i++) {
            //add 1e-5 to avoid division by zero
            float _x = static_cast<float>(mu[i].m10 / (mu[i].m00 + 1e-5));
            float _y = static_cast<float>(mu[i].m01 / (mu[i].m00 + 1e-5));
            if (_x == 0.f && _y == 0.f)
                continue;
            else{
                if (contourArea(contours[i]) > 300)
                    points.push_back(Point2f(_x, _y));
                else {
                    circle(matResult, Point2f(_x, _y), 1, Scalar(128, 128, 128), 10, 8, 0);
                /*
                        __android_log_print(ANDROID_LOG_DEBUG, APPNAME,
                        "contours %d, Center point %.2f, %.2f, Area, %f",
                        (int) i, _x, _y, contourArea(contours[i]));
                */
                }
            }

        }
   /*
        for (int i = 0; i < points.size(); i++)
           __android_log_print(ANDROID_LOG_DEBUG, APPNAME, "result points are %d, Center point %.2f, %.2f",
                                (int)i, points[i].x, points[i].y);
    */
    } else {
        __android_log_print(ANDROID_LOG_DEBUG, APPNAME, "There is no contours");
    }
  //  __android_log_print(ANDROID_LOG_DEBUG, APPNAME, "SIZE : points : %d", (int) points.size());
}

void detect_obj::clusturing_kmeans(std::vector<Point2f> points) {
    int clusterCount = 2; // two hands
    Mat labels;
    Mat centers;
    double compactness = kmeans(points, clusterCount, labels,
                                TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
                                3, KMEANS_PP_CENTERS, centers);
}
