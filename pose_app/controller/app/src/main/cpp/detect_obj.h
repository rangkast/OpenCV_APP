//
// Created by administrator on 2021. 12. 21..
//

#ifndef CONTROLLER_DETECT_OBJ_H
#define CONTROLLER_DETECT_OBJ_H

#include "utils.h"

using namespace cv;

class detect_obj {

public:
    void detect_points(pt_camera_info info, Mat &matInput, Mat &matResult, std::vector<Point2f> &points);
    void clusturing_kmeans(std::vector<Point2f> points);
};

#endif //CONTROLLER_DETECT_OBJ_H
