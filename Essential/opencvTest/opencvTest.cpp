#include <iostream>
#include <opencv2/opencv.hpp>

void cameraDisplacement(const cv::Mat& rvec1, const cv::Mat& tvec1, const cv::Mat& rvec2, const cv::Mat& tvec2,
  cv::Mat& rvec1to2, cv::Mat& tvec1to2) {
  cv::Mat R1, R2, R1to2;
  cv::Rodrigues(rvec1, R1);
  cv::Rodrigues(rvec2, R2);
  R1to2 = R2 * R1.t();
  cv::Rodrigues(R1to2, rvec1to2);

  tvec1to2 = -R1to2*tvec1 + tvec2;
}

void compute_R_t_fromEssentialMatrix(const cv::Mat& E, std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& ts, std::vector<cv::Mat>& ts2) {
  //https://github.com/libmv/libmv/blob/8040c0f6fa8e03547fd4fbfdfaf6d8ffd5d1988b/src/libmv/multiview/fundamental.cc#L302-L338
  cv::Mat w, u, vt;
  cv::SVDecomp(E, w, u, vt, cv::SVD::FULL_UV);

  // Last column of U is undetermined since d = (a a 0).
  if (cv::determinant(u) < 0) {
    u.col(2) *= -1;
  }

  // Last row of Vt is undetermined since d = (a a 0).
  if (cv::determinant(vt) < 0) {
    vt.row(2) *= -1;
  }
  //std::cout << "vt:\n" << vt << std::endl;

  cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1, 0,
    1, 0, 0,
    0, 0, 1);

  cv::Mat U_W_Vt = u * W * vt;
  cv::Mat U_Wt_Vt = u * W.t() * vt;

  rvecs.resize(4);
  cv::Mat R = U_W_Vt, rvec;
  cv::Rodrigues(R, rvec);
  rvecs[0] = rvec;
  rvecs[1] = rvec;

  cv::Mat R2 = U_Wt_Vt, rvec2;
  cv::Rodrigues(R2, rvec2);
  rvecs[2] = rvec2;
  rvecs[3] = rvec2;

  ts.resize(4);
  ts[0] = u.col(2);
  ts[1] = -u.col(2);
  ts[2] = u.col(2);
  ts[3] = -u.col(2);

  //https://en.wikipedia.org/wiki/Essential_matrix#Determining_R_and_t_from_E
  ts2.resize(4);
  cv::Mat Z = (cv::Mat_<double>(3, 3) << 0, 1, 0,
    -1, 0, 0,
    0, 0, 0);
  cv::Mat tskew = u*Z*u.t();
  ts2[0] = (cv::Mat_<double>(3, 1) << tskew.at<double>(2, 1),
    tskew.at<double>(0, 2),
    tskew.at<double>(1, 0));
  ts2[1] = -ts[0];
  ts2[2] = ts[0];
  ts2[3] = -ts[0];
}

void transform(const cv::Point3d& pt, const cv::Mat& rvec, const cv::Mat& tvec, cv::Point3d& ptTrans) {
  cv::Mat R;
  cv::Rodrigues(rvec, R);

  cv::Mat matPt = (cv::Mat_<double>(3, 1) << pt.x, pt.y, pt.z);
  cv::Mat matPtTrans = R * matPt + tvec;
  ptTrans.x = matPtTrans.at<double>(0, 0);
  ptTrans.y = matPtTrans.at<double>(1, 0);
  ptTrans.z = matPtTrans.at<double>(2, 0);
}

void recoverPoseFromPnP(const std::vector<cv::Point3d>& objectPoints1, const cv::Mat& rvec1, const cv::Mat& tvec1, const std::vector<cv::Point2d>& imagePoints2,
                        const cv::Mat& cameraMatrix, cv::Mat& rvec1to2, cv::Mat& tvec1to2) {
  cv::Mat R1;
  cv::Rodrigues(rvec1, R1);

  //transform object points in camera frame
  std::vector<cv::Point3d> objectPoints1InCam;
  for (size_t i = 0; i < objectPoints1.size(); i++) {
    cv::Point3d ptTrans;
    transform(objectPoints1[i], rvec1, tvec1, ptTrans);
    objectPoints1InCam.push_back(ptTrans);
  }

  cv::solvePnP(objectPoints1InCam, imagePoints2, cameraMatrix, cv::noArray(), rvec1to2, tvec1to2, false, cv::SOLVEPNP_EPNP);
}

int main() {
  //object points
  std::vector<cv::Point3d> objectPoints;

  //4 planar points
  objectPoints.push_back(cv::Point3d(-0.5, -0.5, 0.0));
  objectPoints.push_back(cv::Point3d(0.5, -0.5, 0.0));
  objectPoints.push_back(cv::Point3d(0.5, 0.5, 0.0));
  objectPoints.push_back(cv::Point3d(-0.5, 0.5, 0.0));

  //4 more points
  objectPoints.push_back(cv::Point3d(-0.75, 0.22, 0.84));
  objectPoints.push_back(cv::Point3d(0.11, -0.67, 1.53));
  objectPoints.push_back(cv::Point3d(0.39, 0.08, 0.38));
  objectPoints.push_back(cv::Point3d(-0.45, -0.18, 0.23));

  //initial camera pose
  std::cout << "CV_PI\n" << CV_PI << std::endl;
  cv::Mat rvec1 = (cv::Mat_<double>(3, 1) << 5.0*CV_PI / 180.0, -3.0*CV_PI / 180.0, 8.0*CV_PI / 180.0);
  cv::Mat tvec1 = (cv::Mat_<double>(3, 1) << 0.1, 0.2, 2.0);
  std::cout << "rvec1: " << rvec1.t() << std::endl;
  std::cout << "tvec1: " << tvec1.t() << std::endl;

  //project points
  cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
    600.0, 0.0, 320.0,
    0.0, 600.0, 240.0,
    0.0, 0.0, 1.0);

  cv::Mat distCoeffs = (cv::Mat_<double>(4, 1) << 
    0.072867,
    -0.026268,
    0.007135,
    -0.000997);
    
  std::vector<cv::Point2d> imagePoints1;
  //ProjectPoints
  //여기 왜 distcoeff가 없지?
  cv::projectPoints(objectPoints, rvec1, tvec1, cameraMatrix, cv::noArray(), imagePoints1);
  for (size_t i = 0; i < imagePoints1.size(); i++) {
    std::cout << "imagePoints1[" << i << "]: " << imagePoints1[i] << std::endl;
  }

  //new camera pose
  cv::Mat rvec2 = (cv::Mat_<double>(3, 1) << -3.0*CV_PI / 180.0, -5.0*CV_PI / 180.0, -3.0*CV_PI / 180.0);
  cv::Mat tvec2 = (cv::Mat_<double>(3, 1) <<  0.1, 0.2, 2.0);
  std::cout << "rvec2: " << rvec2.t() << std::endl;
  std::cout << "tvec2: " << tvec2.t() << std::endl;

  //camera displacement
  //수식으로 두 카메라 포즈 계산
  cv::Mat rvec1to2, tvec1to2;
  cameraDisplacement(rvec1, tvec1, rvec2, tvec2, rvec1to2, tvec1to2);
  std::cout << "rvec1to2: " << rvec1to2.t() << std::endl;
  std::cout << "tvec1to2: " << tvec1to2.t() << std::endl;

  //project points using new camera pose
  std::vector<cv::Point2d> imagePoints2;
  cv::projectPoints(objectPoints, rvec2, tvec2, cameraMatrix, cv::noArray(), imagePoints2);
  for (size_t i = 0; i < imagePoints2.size(); i++) {
    std::cout << "imagePoints2[" << i << "]: " << imagePoints2[i] << std::endl;
  }

  //find fundamental matrix
  cv::Mat F = cv::findFundamentalMat(imagePoints1, imagePoints2, cv::noArray(), cv::FM_7POINT);
  std::cout << "\nF:\n" << F << std::endl;

  //calculate error
  double meanError = 0.0;
  for (size_t i = 0; i < objectPoints.size(); i++) {
    cv::Point2d x1 = imagePoints1[i], x2 = imagePoints2[i];
    double F0 = F.at<double>(0, 0) * x1.x + F.at<double>(0, 1) * x1.y + F.at<double>(0, 2);
    double F1 = F.at<double>(1, 0) * x1.x + F.at<double>(1, 1) * x1.y + F.at<double>(1, 2);
    double F2 = F.at<double>(2, 0) * x1.x + F.at<double>(2, 1) * x1.y + F.at<double>(2, 2);
    meanError += x2.x * F0 + x2.y * F1 + F2;
  }
  std::cout << "meanError: " << meanError / objectPoints.size() << std::endl;

  //essential matrix
  cv::Mat E = cameraMatrix.t() * F * cameraMatrix;
  std::cout << "\nE:\n" << E << std::endl;

  cv::Mat E2 = cv::findEssentialMat(imagePoints1, imagePoints2, cameraMatrix, cv::LMEDS);
  std::cout << "E2:\n" << E2 << std::endl;
  double scale = E.at<double>(2, 2) / E2.at<double>(2, 2);
  std::cout << "E2:\n" << scale*E2 << std::endl;

  //recover pose
  cv::Mat R, t, rvec;
  cv::recoverPose(E, imagePoints1, imagePoints2, cameraMatrix, R, t);
  cv::Rodrigues(R, rvec);
  std::cout << "\nrvec1to2: " << rvec1to2.t() << std::endl;
  std::cout << "rvec (from recoverPose): " << rvec.t() << std::endl;
  std::cout << "\ntvec1to2: " << tvec1to2.t() << std::endl;
  std::cout << "t (from recoverPose): " << t.t() << std::endl;

  double scalePose = tvec1to2.at<double>(2, 0) / t.at<double>(2, 0);
  std::cout << " tvec1to2.at<double>(2, 0): " <<  tvec1to2.at<double>(2, 0) << std::endl;
  std::cout << "t (from recoverPose after scale): " << scalePose*t.t() << std::endl;

  //manual recover pose
  std::cout << "\ncompute_R_t_fromEssentialMatrix:" << std::endl;
  std::vector<cv::Mat> rvecs, tvecs, tvecs2;
  compute_R_t_fromEssentialMatrix(E, rvecs, tvecs, tvecs2);
  for (int i = 0; i < 4; i++) {
    std::cout << "\nrvecs[" << i << "]: " << rvecs[i].t() << std::endl;
    std::cout << "tvecs[" << i << "]: " << tvecs[i].t() << std::endl;
    scalePose = tvec1to2.at<double>(2, 0) / tvecs[i].at<double>(2, 0);
    std::cout << "tvecs[" << i << "] (after scale): " << scalePose * tvecs[i].t() << std::endl;

    scalePose = tvec1to2.at<double>(2, 0) / tvecs2[i].at<double>(2, 0);
    std::cout << "tvecs2[" << i << "] (after scale): " << scalePose * tvecs2[i].t() << std::endl;
  }

  //pose from solvePnP and known camera pose for image1
  cv::Mat rvec1to2_pnp, tvec1to2_pnp;
  recoverPoseFromPnP(objectPoints, rvec1, tvec1, imagePoints2, cameraMatrix, rvec1to2_pnp, tvec1to2_pnp);
  std::cout << "\nRecover pose using PnP:" << std::endl;
  std::cout << "rvec1to2: " << rvec1to2.t() << std::endl;
  std::cout << "rvec1to2_pnp: " << rvec1to2_pnp.t() << std::endl;
  std::cout << "tvec1to2: " << tvec1to2.t() << std::endl;
  std::cout << "tvec1to2_pnp: " << tvec1to2_pnp.t() << std::endl;

  return 0;
}