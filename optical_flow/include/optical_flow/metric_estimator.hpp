#pragma once

#include <numeric>
#include <vector>
#include "optical_flow/bottom_calibration.hpp"

namespace auv_flow
{
struct MetricSettings
{
  int min_tracks{12};
  double min_inlier_ratio{0.6}, residual_threshold_m{0.025};
  double pixel_noise_px{0.5}, velocity_noise_mps{0.02};
  double height_std_m{0}, focal_relative_std{0.01}, principal_point_std_px{0.5};
  double extrinsic_angle_std_rad{0.01}, lever_arm_std_m{0.005};
  double unobserved_variance{1e6}, max_speed_mps{5.0}, min_ray_cosine{0.2};

  void validate() const
  {
    for (double value : {min_inlier_ratio, residual_threshold_m, pixel_noise_px,
      velocity_noise_mps, height_std_m, focal_relative_std, principal_point_std_px,
      extrinsic_angle_std_rad, lever_arm_std_m, unobserved_variance, max_speed_mps,
      min_ray_cosine})
    {
      if (!std::isfinite(value) || value <= 0) {
        throw std::runtime_error("Metric scale and uncertainty settings must be finite and positive");
      }
    }
    if (min_tracks < 4 || min_inlier_ratio <= 0.5 || min_inlier_ratio > 1 ||
      min_ray_cosine >= 1 || unobserved_variance < 1e4)
    {
      throw std::runtime_error("Invalid robust-estimation or unobserved-axis settings");
    }
  }
};

struct MetricEstimate
{
  cv::Vec3d velocity_body;
  cv::Matx33d covariance_body;
  std::vector<size_t> inliers;
  double residual_rms_m{0};
};

inline cv::Vec3d floorOffset(const cv::Point2f & pixel, const cv::Matx33d & inverse_k,
  const cv::Matx33d & r_wc, double height, double min_cosine)
{
  const auto ray = r_wc * (inverse_k * cv::Vec3d(pixel.x, pixel.y, 1));
  if (!finite(ray) || !std::isfinite(height) || height <= 0 ||
    ray[2] / cv::norm(ray) >= -min_cosine)
  {
    throw std::runtime_error("Camera ray does not safely intersect the horizontal floor");
  }
  return ray * (-height / ray[2]);
}

inline std::vector<cv::Vec2d> metricDisplacements(
  const std::vector<cv::Point2f> & from, const std::vector<cv::Point2f> & to,
  const BottomCalibration & c, const cv::Matx33d & r0, const cv::Matx33d & r1,
  double body_height, double min_cosine)
{
  const cv::Matx33d inverse_k = c.k.inv();
  const auto lever_motion = (r1-r0)*c.t_bc;
  const double h0 = body_height + (r0*c.t_bc)[2];
  const double h1 = body_height + (r1*c.t_bc)[2];
  std::vector<cv::Vec2d> result;
  result.reserve(from.size());
  for (size_t i = 0; i < from.size(); ++i) {
    const auto a = floorOffset(from[i], inverse_k, r0*c.r_bc, h0, min_cosine);
    const auto b = floorOffset(to[i], inverse_k, r1*c.r_bc, h1, min_cosine);
    const auto displacement = a-b-lever_motion;
    result.emplace_back(displacement[0], displacement[1]);
  }
  return result;
}

inline cv::Vec2d meanDisplacement(const std::vector<cv::Vec2d> & d,
  const std::vector<size_t> & indices)
{
  cv::Vec2d sum(0, 0);
  for (const auto i : indices) {sum += d[i];}
  return sum / static_cast<double>(indices.size());
}

inline MetricEstimate estimateMetric(const std::vector<cv::Point2f> & from,
  const std::vector<cv::Point2f> & to, double dt, const Orientation & o0,
  const Orientation & o1, double body_height, const BottomCalibration & calibration,
  const MetricSettings & settings)
{
  calibration.validate();
  settings.validate();
  if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(body_height) || body_height <= 0 ||
    from.size() != to.size() || from.size() < static_cast<size_t>(settings.min_tracks) ||
    o0.stamp_ns <= 0 || o1.stamp_ns <= o0.stamp_ns ||
    std::abs((o1.stamp_ns-o0.stamp_ns)*1e-9-dt) > 1e-6 ||
    !std::isfinite(o0.variance) || !std::isfinite(o1.variance) ||
    o0.variance <= 0 || o1.variance <= 0)
  {
    throw std::runtime_error("Unknown metric scale, orientation, timestamps or insufficient tracks");
  }
  for (size_t i = 0; i < from.size(); ++i) {
    for (const auto & p : {from[i], to[i]}) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.y < 0 ||
        p.x >= calibration.size.width || p.y >= calibration.size.height)
      {
        throw std::runtime_error("Track outside calibrated image");
      }
    }
  }
  const auto r0 = rotation(o0.q), r1 = rotation(o1.q);
  const auto d = metricDisplacements(from, to, calibration, r0, r1,
    body_height, settings.min_ray_cosine);
  const double threshold = std::max(settings.residual_threshold_m,
    3*settings.pixel_noise_px*(body_height+cv::norm(calibration.t_bc)) /
    std::min(calibration.k(0, 0), calibration.k(1, 1)));
  // Deterministic translation consensus in metres AFTER rotation compensation.
  std::vector<size_t> keep;
  const size_t hypotheses = std::min<size_t>(100, d.size());
  for (size_t h = 0; h < hypotheses; ++h) {
    const auto candidate = d[h*d.size()/hypotheses];
    std::vector<size_t> support;
    for (size_t i = 0; i < d.size(); ++i) {
      if (cv::norm(d[i]-candidate) < threshold) {support.push_back(i);}
    }
    if (support.size() > keep.size()) {keep = std::move(support);}
  }
  for (int iteration = 0; iteration < 3; ++iteration) {
    if (keep.empty()) {throw std::runtime_error("No floor-motion consensus");}
    const auto centre = meanDisplacement(d, keep);
    keep.clear();
    for (size_t i = 0; i < d.size(); ++i) {
      if (cv::norm(d[i]-centre) < threshold) {keep.push_back(i);}
    }
  }
  if (keep.size() < static_cast<size_t>(settings.min_tracks) ||
    static_cast<double>(keep.size())/d.size() < settings.min_inlier_ratio)
  {
    throw std::runtime_error("Insufficient consistent floor-motion inliers");
  }
  cv::Vec2d pixel_mean(0, 0);
  for (const auto i : keep) {pixel_mean += cv::Vec2d(from[i].x, from[i].y);}
  pixel_mean /= static_cast<double>(keep.size());
  cv::Matx22d scatter = cv::Matx22d::zeros();
  for (const auto i : keep) {
    const cv::Vec2d offset = cv::Vec2d(from[i].x, from[i].y)-pixel_mean;
    scatter += offset*offset.t();
  }
  scatter *= 1.0/(keep.size()-1);
  cv::Vec2d eigenvalues;
  cv::eigen(scatter, eigenvalues);
  if (eigenvalues[1] < 25) {
    throw std::runtime_error("Features lack two-dimensional spatial support");
  }
  const auto centre = meanDisplacement(d, keep);
  if (cv::norm(centre)/dt > settings.max_speed_mps) {
    throw std::runtime_error("Metric speed exceeds configured physical limit");
  }
  cv::Matx22d covariance = cv::Matx22d::zeros();
  double squared_residuals = 0;
  for (const auto i : keep) {
    const auto residual = d[i]-centre;
    covariance += residual*residual.t();
    squared_residuals += residual.dot(residual);
  }
  covariance *= 1.0/(keep.size()*(keep.size()-1));
  const double pixel_scale = settings.pixel_noise_px*(body_height+cv::norm(calibration.t_bc)) /
    std::min(calibration.k(0, 0), calibration.k(1, 1));
  covariance += cv::Matx22d::eye()*(2*pixel_scale*pixel_scale/keep.size());
  // Numerical propagation: systematic calibration/scale errors do not shrink with N.
  const auto add_variance = [&](const cv::Vec2d & shifted, double epsilon, double variance) {
      const auto jac = (shifted-centre)/epsilon;
      covariance += (jac*jac.t())*variance;
    };
  const auto mean_for = [&](const BottomCalibration & c, const cv::Matx33d & a,
      const cv::Matx33d & b, double height) {
      return meanDisplacement(metricDisplacements(from, to, c, a, b, height,
        settings.min_ray_cosine), keep);
    };
  constexpr double epsilon = 1e-5;
  add_variance(mean_for(calibration, r0, r1, body_height+epsilon), epsilon,
    settings.height_std_m*settings.height_std_m);
  for (int axis = 0; axis < 3; ++axis) {
    add_variance(mean_for(calibration, perturb(r0, axis, epsilon), r1, body_height),
      epsilon, o0.variance);
    add_variance(mean_for(calibration, r0, perturb(r1, axis, epsilon), body_height),
      epsilon, o1.variance);
    auto c = calibration;
    c.r_bc = perturb(c.r_bc, axis, epsilon);
    add_variance(mean_for(c, r0, r1, body_height), epsilon,
      settings.extrinsic_angle_std_rad*settings.extrinsic_angle_std_rad);
    c = calibration;
    c.t_bc[axis] += epsilon;
    add_variance(mean_for(c, r0, r1, body_height), epsilon,
      settings.lever_arm_std_m*settings.lever_arm_std_m);
  }
  for (int axis = 0; axis < 2; ++axis) {
    auto c = calibration;
    c.k(axis, axis) += epsilon;
    const double focal_std = settings.focal_relative_std*calibration.k(axis, axis);
    add_variance(mean_for(c, r0, r1, body_height), epsilon, focal_std*focal_std);
    c = calibration;
    c.k(axis, 2) += epsilon;
    add_variance(mean_for(c, r0, r1, body_height), epsilon,
      settings.principal_point_std_px*settings.principal_point_std_px);
  }
  covariance = covariance*(1.0/(dt*dt)) +
    cv::Matx22d::eye()*(settings.velocity_noise_mps*settings.velocity_noise_mps);
  const cv::Matx33d world_cov(covariance(0, 0), covariance(0, 1), 0,
    covariance(1, 0), covariance(1, 1), 0, 0, 0, settings.unobserved_variance);
  const cv::Vec3d velocity = r1.t()*cv::Vec3d(centre[0]/dt, centre[1]/dt, 0);
  auto body_cov = r1.t()*world_cov*r1;
  const cv::Matx33d skew(0, -velocity[2], velocity[1], velocity[2], 0, -velocity[0],
    -velocity[1], velocity[0], 0);
  body_cov += (skew*skew.t())*o1.variance;
  body_cov = (body_cov+body_cov.t())*0.5;
  if (!finite(velocity) || !finite(body_cov)) {
    throw std::runtime_error("Nonfinite metric estimate or covariance");
  }
  return {velocity, body_cov, keep, std::sqrt(squared_residuals/keep.size())};
}
}  // namespace auv_flow
