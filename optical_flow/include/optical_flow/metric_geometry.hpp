#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iterator>
#include <stdexcept>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace auv_flow
{
inline bool finite(const cv::Matx33d & m) {return cv::checkRange(cv::Mat(m));}
inline bool finite(const cv::Vec3d & v) {return cv::checkRange(cv::Mat(v));}

inline cv::Matx33d rotation(const cv::Vec4d & quaternion)
{
  const double norm = cv::norm(quaternion);
  if (!cv::checkRange(cv::Mat(quaternion)) || std::abs(norm - 1.0) > 0.01) {
    throw std::runtime_error("Expected a finite unit quaternion [x,y,z,w]");
  }
  const auto q = quaternion / norm;
  const double x = q[0], y = q[1], z = q[2], w = q[3];
  return {1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
    2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
    2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)};
}

inline cv::Vec4d slerp(cv::Vec4d a, cv::Vec4d b, double f)
{
  rotation(a);
  rotation(b);
  a /= cv::norm(a);
  b /= cv::norm(b);
  double dot = a.dot(b);
  if (dot < 0) {b = -b; dot = -dot;}
  if (dot > 0.9995) {
    const auto q = a + f * (b-a);
    return q / cv::norm(q);
  }
  const double angle = std::acos(std::clamp(dot, -1.0, 1.0));
  return (std::sin((1-f)*angle)*a + std::sin(f*angle)*b) / std::sin(angle);
}

inline cv::Matx33d perturb(const cv::Matx33d & r, int axis, double epsilon)
{
  cv::Vec3d v(0, 0, 0);
  v[axis] = epsilon;
  cv::Matx33d delta;
  cv::Rodrigues(v, delta);
  return delta * r;
}

struct Orientation
{
  int64_t stamp_ns{0};
  cv::Vec4d q{0, 0, 0, 1};  // body -> fixed reference frame
  double variance{0};      // upper bound on orientation covariance eigenvalues, rad^2
};

// Explicit test assumption only: body axes aligned with the fixed reference.
inline Orientation assumedLevelOrientation(int64_t stamp_ns, double std_rad)
{
  if (stamp_ns <= 0 || !std::isfinite(std_rad) || std_rad <= 0 ||
    !std::isfinite(std_rad*std_rad) || std_rad*std_rad <= 0)
  {
    throw std::runtime_error("Invalid fixed-orientation test timestamp or uncertainty");
  }
  return {stamp_ns, {0, 0, 0, 1}, std_rad*std_rad};
}

class OrientationBuffer
{
public:
  explicit OrientationBuffer(double max_gap = 0.1) : max_gap_(max_gap)
  {
    if (!std::isfinite(max_gap) || max_gap <= 0) {
      throw std::runtime_error("Invalid orientation synchronization gap");
    }
  }
  void clear() {samples_.clear();}
  void add(const Orientation & sample)
  {
    rotation(sample.q);
    if (sample.stamp_ns <= 0 || !std::isfinite(sample.variance) || sample.variance <= 0) {
      throw std::runtime_error("Unknown orientation timestamp or uncertainty");
    }
    Orientation normalized = sample;
    normalized.q /= cv::norm(normalized.q);
    const auto it = std::lower_bound(samples_.begin(), samples_.end(), sample.stamp_ns,
      [](const Orientation & o, int64_t t) {return o.stamp_ns < t;});
    if (it != samples_.end() && it->stamp_ns == sample.stamp_ns) {*it = normalized;} else {
      samples_.insert(it, normalized);
    }
    while (samples_.size() > 2000) {samples_.pop_front();}
  }
  Orientation sample(int64_t stamp) const
  {
    const auto it = std::lower_bound(samples_.begin(), samples_.end(), stamp,
      [](const Orientation & o, int64_t t) {return o.stamp_ns < t;});
    if (it != samples_.end() && it->stamp_ns == stamp) {return *it;}
    if (it == samples_.begin() || it == samples_.end()) {
      throw std::runtime_error("No bracketing orientation at image acquisition time");
    }
    const auto & a = *std::prev(it);
    const auto & b = *it;
    if ((b.stamp_ns - a.stamp_ns)*1e-9 > max_gap_) {
      throw std::runtime_error("Orientation samples have an excessive timestamp gap");
    }
    const double f = static_cast<double>(stamp-a.stamp_ns)/(b.stamp_ns-a.stamp_ns);
    return {stamp, slerp(a.q, b.q, f), std::max(a.variance, b.variance)};
  }
private:
  double max_gap_;
  std::deque<Orientation> samples_;
};
}  // namespace auv_flow
