#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <math.h>

#include "point_tracker.h"

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

//float *temp_array;
std::vector<Point3f> model_points;
std::vector<Point2f> object_points;

Point2f point_tracker::avg_coord(std::vector<Point2f> getPoints) {

    float avg_x = 0;
    float avg_y = 0;
    int size = getPoints.size();
    for (int i = 0; i < size; i++) {
        avg_x += getPoints[i].x;
        avg_y += getPoints[i].y;
    }
    avg_x = avg_x / size;
    avg_y = avg_y / size;

    return Point2f(avg_x, avg_y);
}

//거리와 좌표로 compare
bool point_comparator(pair<Point2f, float> a, pair<Point2f, float> b) {
    if (a.second == b.second) {
        if (b.first.x > a.first.x) {
            return true;
        } else if (b.first.x == a.first.x) {
            if (b.first.y > a.first.y)
                return true;
            else
                return false;
        } else
            return false;
    } else {
        return b.second > a.second;
    }
}

void point_tracker::refresh_model_points(std::vector<Point2f> getPoints) {
    model_points.clear();
    object_points.clear();
    __android_log_print(ANDROID_LOG_DEBUG, APPNAME, "refresh model points");
    // 3D model points.
    for (int i = 0; i < getPoints.size(); i++) {
        model_points.push_back(Point3f(getPoints[i].x, getPoints[i].y, 10.0f));
        object_points.push_back(Point2f(getPoints[i].x, getPoints[i].y));
    }
}

std::vector<pair<Point2f, float>> point_tracker::add_distance(std::vector<Point2f> srcPoints, Point2f mid_point) {
    int size = srcPoints.size();
    std::vector<pair<Point2f, float>> points;
    for (int i = 0; i < size; i++) {
       float distance = sqrt(pow(mid_point.x - srcPoints[i].x, 2) + pow(mid_point.y - srcPoints[i].y, 2));
        points.push_back(make_pair(Point2f(srcPoints[i].x, srcPoints[i].y), distance));
    }
    return points;
}

bool point_tracker::estimate_pose(pt_camera_info info, InputOutputArray img, std::vector<Point2f> getPoints, Mat* Rot, Mat* r, Mat* t) {

    int m_size = model_points.size();
    int g_size = getPoints.size();

 //   __android_log_print(ANDROID_LOG_DEBUG, APPNAME, "size: %d %d", g_size, m_size);
  //  __android_log_print(ANDROID_LOG_DEBUG, APPNAME, "%s %d %d %f %d", "estimate_pose", info.res_x, info.res_y, info.fov, info.touch_status);

    Point2f mid_point = avg_coord(getPoints);
    std::vector<pair<Point2f, float>> points_dis;

    points_dis = add_distance(getPoints, mid_point);

    //sort
    /*
    for (int i = 0; i < g_size; i++) {
        __android_log_print(ANDROID_LOG_DEBUG, APPNAME,
                            "get points are %d, Center point %.2f, %.2f : %.4f",
                            (int) i, points_dis[i].first.x, points_dis[i].first.y, points_dis[i].second);
    }
*/
    sort(points_dis.begin(), points_dis.end(), point_comparator);

/*
    for (int i = 0; i < g_size; i++) {
        __android_log_print(ANDROID_LOG_DEBUG, APPNAME,
                            "sort points are %d, Center point %.2f, %.2f : %.4f",
                            (int) i, points_dis[i].first.x, points_dis[i].first.y, points_dis[i].second);
    }
*/

    std::vector<Point2f> image_points;

    for (int i = 0; i < g_size; i++) {
        image_points.push_back(Point2f(points_dis[i].first.x, points_dis[i].first.y) );
    }

    if (info.touch_status == 0) {
        refresh_model_points(image_points);
        info.start_tracking = 0;
        return true;
    }

    if (g_size <= 2) {
        __android_log_print(ANDROID_LOG_DEBUG, APPNAME, "size not enough");
    } else {
        int diff = g_size - m_size;
        if (diff != 0) {
            refresh_model_points(image_points);
        }

        //Camera internals
        int flags = SOLVEPNP_SQPNP;
        double focal_length = info.res_y; // Approximate focal length.
        Point2f center = Point2f(info.res_x / 2, info.res_y / 2);
        Mat camera_matrix = (Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
        cv::Mat dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion

        Mat irvec = Mat::zeros(3, 1, CV_64FC1);
        Mat itvec = Mat::zeros(3, 1, CV_64FC1);
        std::vector<Mat> irvecs, itvecs;

        // Solve for pose
        //https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga624af8a6641b9bdb487f63f694e8bb90
        info.solutions = solvePnPGeneric(model_points, image_points, camera_matrix, dist_coeffs,
                                        irvecs, itvecs, false, (SolvePnPMethod)flags, irvec, itvec);

        std::vector<Point3f> end_point3f;
        std::vector<Point2f> end_point2f;
        end_point3f.push_back(Point3f(0, 0,100.0));

        projectPoints(end_point3f, irvec, itvec, camera_matrix, dist_coeffs, end_point2f);
        line(img, mid_point, end_point2f[0], Scalar(0,0,255), 1);

        for (int i = 0; i < g_size; i++) {
            line(img, mid_point, image_points[i], Scalar(255,255,255), 1);
            line(img, image_points[i], object_points[i], Scalar(255,0,255), 1);
            putText(img, std::to_string(i + 1), Point2f(image_points[i].x + 10 ,image_points[i].y + 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,255,255), 2);
        }
/*
        __android_log_print(ANDROID_LOG_DEBUG, APPNAME, "%f %f %f %f %f %f", avg_obj.x , avg_obj.y,
                            avg_img.x, avg_img.y,
                            end_point2f[0].x, end_point2f[0].y
        );
*/
    //    __android_log_print(ANDROID_LOG_DEBUG, APPNAME, "%s -> %d", "solvePnPGeneric", info.solutions);

        std::ostringstream mat_data3;
        if (info.solutions > 0) {
            /* debugging
            mat_data3 << "sizeof solution " << info.solutions << "\n";
            mat_data3 << "sizeof rvecs " << irvecs.size() << "\n";
        for (int i = 0; i < info.solutions ; i++){
            double* debugRvec = irvecs[i].ptr<double>();
            mat_data3 << "solvePnPGeneric, irvecs, " << i << ", " << debugRvec[0] << ", " << debugRvec[1] << ", " << debugRvec[2] <<"\n";
            mat_data3 << "\n-------\n";
        }
        for (int i = 0; i < info.solutions ; i++){
            mat_data3 << "itvecs, " << cv::format(itvecs[i], cv::Formatter::FMT_C);
            mat_data3 << "\n-------\n";
        }
        __android_log_print(ANDROID_LOG_DEBUG, APPNAME, "%s", mat_data3.str().c_str());
*/

        int rdepth = irvec.empty() ? CV_64F : irvec.depth();
        int tdepth = itvec.empty() ? CV_64F : itvec.depth();
 //       Mat offset(3, 1, CV_64F, new float[]{0.5f, 0.5f, 0.5f});
        if (info.start_tracking == 1) {
            //중간점 잡으려고 했으나 별로....
//            irvecs[0] = (irvecs[0] + info.rvec) * offset;
  //          itvecs[0] = (itvecs[0] + info.tvec) * offset;
            irvecs[0].convertTo(irvec, rdepth);
            itvecs[0].convertTo(itvec, tdepth);
        } else {
            irvecs[0].convertTo(irvec, rdepth);
            itvecs[0].convertTo(itvec, tdepth);
        }
        info.rvec = irvec.clone();
        info.tvec = itvec.clone();


        //여기 모르겠음.
        //irvec = -irvec;

        //https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac
        //https://darkpgmr.tistory.com/99
        Rodrigues(irvec, info.R);
        // debugging
        info.rv = info.R.ptr<double>();
/*
        __android_log_print(ANDROID_LOG_DEBUG, APPNAME,
                            "rotation vector: %.3f %.3f %.3f, %.3f %.3f %.3f, %.3f %.3f %.3f",
                            info.rv[0],info.rv[1],info.rv[2],info.rv[3],info.rv[4],info.rv[5],info.rv[6],info.rv[7],info.rv[8]);
*/
        info.tv = info.tvec.ptr<double>();
/*
        __android_log_print(ANDROID_LOG_DEBUG, APPNAME,
                            "trans vector: %.3f %.3f %.3f",info.tv[0],info.tv[1],info.tv[2]);
*/
        //여기 몰겠음 그냥 OpenGL 해상도 맞게 하려는 듯..
        Mat factor(3, 1, CV_64F, new double[]{1, 1, 1});  // Offset value to move the translation vector
        Mat restvec = itvec + info.R*factor;    // Move tVec to refer to the center of the paper
        restvec = restvec / 100;
        // TODO : Check Converting pixel coordinates to OpenGL world coordinates
        /*
        double* rtv = itvec.ptr<double>();
        __android_log_print(ANDROID_LOG_DEBUG, APPNAME,
                            "itvec vector: %.3f %.3f %.3f",rtv[0],rtv[1],rtv[2]);
*/
        *t = restvec.clone();
        *r = irvec.clone();
        }
    }
    info.start_tracking = 1;
    *Rot = info.R.clone();
    return true;
}

