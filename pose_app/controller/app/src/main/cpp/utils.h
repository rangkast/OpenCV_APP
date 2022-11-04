//
// Created by administrator on 2021. 12. 21..
//

#ifndef CONTROLLER_UTILS_H
#define CONTROLLER_UTILS_H

#include <opencv2/opencv.hpp>

#define APPNAME "opencv"

// Definition
using f = float;
template<int n> using vec = cv::Vec<f, n>;
using vec2 = vec<2>;
using vec3 = vec<3>;
template<int y, int x> using mat = cv::Matx<f, y, x>;
//

struct pt_camera_info final
{
    using f = std::float_t;
    f fov = 0;
    f fps = 0;
    int res_x = 0;
    int res_y = 0;
    int touch_status = 1;
    int start_tracking = 0;
    int contours = 0;
    int solutions = 0;

    cv::String name;
    cv::Mat R;
    cv::Mat rvec;
    cv::Mat tvec;

    double* rv;
    double* tv;
};

#endif //CONTROLLER_UTILS_H
