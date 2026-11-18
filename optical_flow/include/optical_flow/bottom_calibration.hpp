#pragma once

#include <string>
#include "optical_flow/metric_geometry.hpp"

namespace auv_flow
{
struct BottomCalibration
{
  cv::Size size;
  cv::Matx33d k{cv::Matx33d::zeros()};  // rectified pinhole matrix (left 3x3 of P)
  cv::Matx33d r_bc{cv::Matx33d::eye()}; // camera optical -> body
  cv::Vec3d t_bc{0, 0, 0};            // camera centre in body coordinates, metres
  std::string camera_frame, body_frame;
  bool calibrated{false}, extrinsics_calibrated{false};

  void validate() const
  {
    if (!calibrated || !extrinsics_calibrated || size.width <= 0 || size.height <= 0 ||
      camera_frame.empty() || body_frame.empty() || !finite(k) ||
      k(0, 0) <= 0 || k(1, 1) <= 0 || std::abs(cv::determinant(k)) < 1e-12 ||
      std::abs(k(2, 0)) > 1e-12 || std::abs(k(2, 1)) > 1e-12 ||
      std::abs(k(2, 2)-1) > 1e-12 || !finite(r_bc) || !finite(t_bc) ||
      cv::norm(r_bc.t()*r_bc-cv::Matx33d::eye()) > 1e-6 ||
      std::abs(cv::determinant(r_bc)-1) > 1e-6)
    {
      throw std::runtime_error("Bottom intrinsics/extrinsics are missing, unconfirmed or invalid");
    }
  }

  static BottomCalibration load(const std::string & path)
  {
    cv::FileStorage file(path, cv::FileStorage::READ);
    if (!file.isOpened()) {throw std::runtime_error("Cannot read bottom calibration: " + path);}
    BottomCalibration c;
    int calibrated = 0, extrinsics = 0;
    file["calibrated"] >> calibrated;
    file["extrinsics_calibrated"] >> extrinsics;
    c.calibrated = calibrated == 1;
    c.extrinsics_calibrated = extrinsics == 1;
    file["image_width"] >> c.size.width;
    file["image_height"] >> c.size.height;
    file["camera_frame"] >> c.camera_frame;
    file["body_frame"] >> c.body_frame;
    cv::Mat matrix, quaternion, translation;
    file["rectified_camera_matrix"] >> matrix;
    file["camera_to_body_quaternion_xyzw"] >> quaternion;
    file["camera_position_body_m"] >> translation;
    if (matrix.rows != 3 || matrix.cols != 3 || matrix.type() != CV_64F ||
      quaternion.total() != 4 || quaternion.type() != CV_64F ||
      translation.total() != 3 || translation.type() != CV_64F)
    {
      throw std::runtime_error("Calibration matrices must be double: K 3x3, quaternion 4, position 3");
    }
    c.k = cv::Matx33d(matrix);
    const auto q = quaternion.reshape(1, 4);
    c.r_bc = rotation({q.at<double>(0), q.at<double>(1), q.at<double>(2), q.at<double>(3)});
    const auto t = translation.reshape(1, 3);
    c.t_bc = {t.at<double>(0), t.at<double>(1), t.at<double>(2)};
    return c;
  }
};
}  // namespace auv_flow
