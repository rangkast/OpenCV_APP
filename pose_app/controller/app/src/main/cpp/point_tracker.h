//
// Created by administrator on 2021. 12. 21..
//

#ifndef CONTROLLER_POINT_TRACKER_H
#define CONTROLLER_POINT_TRACKER_H

#include "utils.h"

using namespace cv;
using namespace std;
class point_tracker {
public:
    Mat test(Mat* t);

    bool estimate_pose(pt_camera_info info, InputOutputArray img, std::vector<Point2f> points, Mat* Rot, Mat* r, Mat* t);
    void refresh_model_points(std::vector<Point2f> getPoints);
    Point2f avg_coord(std::vector<Point2f> getPoints);
    std::vector<pair<Point2f, float>>  add_distance(std::vector<Point2f> srcPoints, Point2f mid_point);
};

#endif //CONTROLLER_POINT_TRACKER_H
