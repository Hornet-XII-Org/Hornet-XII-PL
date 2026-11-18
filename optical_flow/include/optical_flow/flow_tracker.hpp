#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace auv_flow
{
struct FlowResult
{
  bool valid{false};
  cv::Point2d displacement{0.0, 0.0};
  std::vector<cv::Point2f> corners;
  std::vector<cv::Point2f> from;
  std::vector<cv::Point2f> to;
};

class FlowTracker
{
public:
  void reset()
  {
    previous_.release();
    points_.clear();
  }

  FlowResult update(const cv::Mat & gray)
  {
    // Expensive feature detection and image pyramids run on a smaller image.
    // Restore all outputs to input-image pixels so calibration stays unchanged.
    cv::Mat small;
    if (gray.cols > 640) {
      cv::resize(gray, small, cv::Size(640,
        std::max(1, static_cast<int>(std::lround(gray.rows * 640.0 / gray.cols)))),
        0, 0, cv::INTER_AREA);
    } else {
      small = gray;
    }
    if (input_size_ != gray.size()) {
      reset();
      input_size_ = gray.size();
    }
    auto result = updateScaled(small);
    const double sx = static_cast<double>(gray.cols) / small.cols;
    const double sy = static_cast<double>(gray.rows) / small.rows;
    result.displacement.x *= sx;
    result.displacement.y *= sy;
    for (auto * points : {&result.corners, &result.from, &result.to}) {
      for (auto & p : *points) {
        p.x *= sx;
        p.y *= sy;
      }
    }
    return result;
  }

private:
  FlowResult updateScaled(const cv::Mat & gray)
  {
    FlowResult result;
    if (!previous_.empty() && previous_.size() == gray.size() && points_.size() >= 8) {
      std::vector<cv::Point2f> next, back;
      std::vector<unsigned char> forward_ok, backward_ok;
      std::vector<float> errors;
      cv::calcOpticalFlowPyrLK(
        previous_, gray, points_, next, forward_ok, errors, cv::Size(31, 31), 4);
      cv::calcOpticalFlowPyrLK(
        gray, previous_, next, back, backward_ok, errors, cv::Size(31, 31), 4);
      std::vector<cv::Point2f> from, to;
      for (size_t i = 0; i < points_.size(); ++i) {
        if (forward_ok[i] && backward_ok[i] &&
          cv::norm(points_[i] - back[i]) <= 1.5 &&
          next[i].x >= 0 && next[i].y >= 0 &&
          next[i].x < gray.cols && next[i].y < gray.rows)
        {
          from.push_back(points_[i]);
          to.push_back(next[i]);
        }
      }
      result.from = from;
      result.to = to;
      result.valid = from.size() >= 8;
      // Diagnostic displacement only. Metric robust fitting happens after ray
      // projection and orientation compensation in metric_estimator.hpp.
      if (result.valid) {
        std::vector<float> dx, dy;
        for (size_t i = 0; i < from.size(); ++i) {
          dx.push_back(to[i].x-from[i].x);
          dy.push_back(to[i].y-from[i].y);
        }
        std::nth_element(dx.begin(), dx.begin()+dx.size()/2, dx.end());
        std::nth_element(dy.begin(), dy.begin()+dy.size()/2, dy.end());
        result.displacement = {dx[dx.size()/2], dy[dy.size()/2]};
      }
    }
    // Retain accepted tracks to avoid corners jumping around every frame.
    points_ = result.to.size() >= 8 ? result.to : std::vector<cv::Point2f>{};
    if (points_.size() < 150) {
      cv::Mat mask(gray.size(), CV_8UC1, cv::Scalar(255));
      for (const auto & p : points_) {
        cv::circle(mask, p, 12, cv::Scalar(0), -1);
      }
      std::vector<cv::Point2f> fresh;
      cv::goodFeaturesToTrack(gray, fresh, 300 - static_cast<int>(points_.size()),
        0.01, 12.0, mask, 7, false);
      points_.insert(points_.end(), fresh.begin(), fresh.end());
    }
    result.corners = points_;
    gray.copyTo(previous_);
    return result;
  }

private:
  cv::Mat previous_;
  std::vector<cv::Point2f> points_;
  cv::Size input_size_;
};

}  // namespace auv_flow
