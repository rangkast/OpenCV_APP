#ifndef CONTROLLER_POINT_OBJ_H
#define CONTROLLER_POINT_OBJ_H

#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <math.h>

#include "utils.h"

using namespace cv;

class point_obj {
    public: Mat gR;
    public: Mat gt;
    public: Mat gr;

    public: std::vector<Point2f> points;

    void init();
    void push_back(Point2f pts);
    void clearPoints();
};


#endif //CONTROLLER_POINT_OBJ_H
