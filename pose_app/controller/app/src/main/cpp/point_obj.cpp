
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <math.h>

#include "point_obj.h"

using namespace cv;

void point_obj::init(){
    gR = Mat::eye(3, 3, CV_64FC1);
    gt = Mat::zeros(3, 1, CV_64FC1);
    gr = Mat::zeros(3, 1, CV_64FC1);
}

void point_obj::push_back(Point2f p){
    points.push_back(p);
}

void point_obj::clearPoints(){
    points.clear();
}