#include <iostream>
#include <limits>
#include <random>
#include <functional>
#include "optical_flow/metric_estimator.hpp"
#include "optical_flow/flow_tracker.hpp"

namespace
{
int checks = 0;
void require(bool value, const std::string & message)
{
  ++checks;
  if (!value) {throw std::runtime_error(message);}
}
void rejected(const std::function<void()> & f, const std::string & name)
{
  bool failed = false;
  try {f();} catch (const std::exception &) {failed = true;}
  require(failed, "Must withhold: " + name);
}
auv_flow::BottomCalibration calibration()
{
  auv_flow::BottomCalibration c;
  c.calibrated = c.extrinsics_calibrated = true;
  c.size = {640, 480};
  c.k = {400, 0, 319.5, 0, 410, 239.5, 0, 0, 1};
  c.r_bc = auv_flow::rotation({1, 0, 0, 0});
  c.t_bc = {0.12, -0.04, -0.10};
  c.camera_frame = "bottom_camera_optical_frame";
  c.body_frame = "base_link";
  return c;
}
cv::Vec4d quaternion(const cv::Vec3d & axis_angle)
{
  const double angle = cv::norm(axis_angle);
  if (angle == 0) {return {0, 0, 0, 1};}
  const auto v = axis_angle*(std::sin(angle/2)/angle);
  return {v[0], v[1], v[2], std::cos(angle/2)};
}
struct Tracks {std::vector<cv::Point2f> from, to;};
Tracks project(const auv_flow::BottomCalibration & c, const auv_flow::Orientation & o0,
  const auv_flow::Orientation & o1, cv::Vec3d move, double height = 2.0)
{
  const auto r0 = auv_flow::rotation(o0.q), r1 = auv_flow::rotation(o1.q);
  const cv::Vec3d position0(0, 0, height), position1 = position0+move;
  Tracks tracks;
  for (int x = -5; x <= 5; ++x) {
    for (int y = -4; y <= 4; ++y) {
      const cv::Vec3d point(x*0.16, y*0.16, 0);
      const auto a = (r0*c.r_bc).t()*(point-position0-r0*c.t_bc);
      const auto b = (r1*c.r_bc).t()*(point-position1-r1*c.t_bc);
      const auto u = c.k*a, v = c.k*b;
      cv::Point2f p(u[0]/u[2], u[1]/u[2]), q(v[0]/v[2], v[1]/v[2]);
      if (a[2] > 0 && b[2] > 0 && p.x > 50 && q.x > 50 && p.y > 50 && q.y > 50 &&
        p.x < 590 && q.x < 590 && p.y < 430 && q.y < 430)
      {
        tracks.from.push_back(p);
        tracks.to.push_back(q);
      }
    }
  }
  return tracks;
}
}  // namespace

int main(int argc, char ** argv)
{
  try {
    auto c = calibration();
    auv_flow::MetricSettings settings;
    settings.height_std_m = 0.01;
    auv_flow::Orientation a{1000000000LL, {0, 0, 0, 1}, 1e-8};
    auv_flow::Orientation b{1100000000LL, {0, 0, 0, 1}, 1e-8};
    const auto fixed = auv_flow::assumedLevelOrientation(a.stamp_ns, 0.02);
    require(fixed.stamp_ns == a.stamp_ns && cv::norm(fixed.q-cv::Vec4d(0, 0, 0, 1)) < 1e-12 &&
      std::abs(fixed.variance-0.0004) < 1e-12, "Fixed test attitude uses image time and positive variance");
    rejected([&]() {auv_flow::assumedLevelOrientation(0, 0.02);}, "unknown fixed test timestamp");
    rejected([&]() {auv_flow::assumedLevelOrientation(a.stamp_ns, 0);}, "confident fixed attitude assumption");
    rejected([&]() {auv_flow::assumedLevelOrientation(a.stamp_ns,
        std::numeric_limits<double>::quiet_NaN());}, "nonfinite fixed attitude uncertainty");
    const cv::Vec3d motion(0.03, -0.015, 0), expected = motion/0.1;
    auto tracks = project(c, a, b, motion);
    auto estimate = auv_flow::estimateMetric(tracks.from, tracks.to, 0.1, a, b, 2, c, settings);
    require(cv::norm(estimate.velocity_body-expected) < 1e-4, "Metric translation/sign/scale");
    std::cout << "Translation error: " << cv::norm(estimate.velocity_body-expected) << " m/s\n";
    auto fast_b = b;
    fast_b.stamp_ns = 1050000000LL;
    const auto fast = auv_flow::estimateMetric(tracks.from, tracks.to, .05, a, fast_b, 2, c, settings);
    require(cv::norm(fast.velocity_body-2*expected) < 2e-4, "Acquisition dt sets m/s");
    require(fast.covariance_body(0, 0) > estimate.covariance_body(0, 0), "Timing affects covariance");
    auto higher = settings;
    higher.height_std_m = 0.2;
    const auto uncertain = auv_flow::estimateMetric(tracks.from, tracks.to, .1, a, b, 2, c, higher);
    require(uncertain.covariance_body(0, 0) > estimate.covariance_body(0, 0),
      "Height uncertainty cannot disappear with many features");
    higher = settings;
    higher.focal_relative_std = 0.1;
    const auto focal_uncertain = auv_flow::estimateMetric(tracks.from, tracks.to, .1, a, b, 2, c, higher);
    require(focal_uncertain.covariance_body(0, 0) > estimate.covariance_body(0, 0),
      "Focal calibration uncertainty must affect velocity covariance");

    const auto stationary_tracks = project(c, a, b, {0, 0, 0});
    const auto stationary = auv_flow::estimateMetric(stationary_tracks.from,
      stationary_tracks.to, .1, a, b, 2, c, settings);
    require(cv::norm(stationary.velocity_body) < 1e-7 && stationary.covariance_body(0, 0) > 0,
      "Observed stationary motion is valid only with positive covariance");

    a.q = quaternion({0.07, -0.04, 0.3});
    b.q = quaternion({0.10, 0.02, 0.36});
    auto rotating = project(c, a, b, {0, 0, 0});
    const auto rotation_only = auv_flow::estimateMetric(rotating.from, rotating.to, .1,
      a, b, 2, c, settings);
    require(cv::norm(rotation_only.velocity_body) < 1e-4,
      "Pure yaw/pitch/roll with a camera lever arm must not create translation");
    std::cout << "Rotation-only apparent speed: " << cv::norm(rotation_only.velocity_body) << " m/s\n";
    tracks = project(c, a, b, motion);
    estimate = auv_flow::estimateMetric(tracks.from, tracks.to, .1, a, b, 2, c, settings);
    const auto expected_body = auv_flow::rotation(b.q).t()*expected;
    require(cv::norm(estimate.velocity_body-expected_body) < 1e-4,
      "Combined rotation+translation must be expressed in current child frame");
    const auto normal_body = auv_flow::rotation(b.q).t()*cv::Vec3d(0, 0, 1);
    require(normal_body.dot(estimate.covariance_body*normal_body) >= 0.99*settings.unobserved_variance,
      "Unobserved vertical direction must retain large covariance after rotation");
    cv::Vec3d eigenvalues;
    cv::eigen(estimate.covariance_body, eigenvalues);
    require(eigenvalues[2] > 0, "Covariance must be positive definite");

    std::mt19937 random(7);
    std::normal_distribution<float> noise(0, .15);
    double squared_error = 0;
    int covered = 0;
    for (int trial = 0; trial < 30; ++trial) {
      auto noisy = tracks;
      for (size_t i = 0; i < noisy.to.size(); ++i) {
        noisy.to[i] += cv::Point2f(noise(random), noise(random));
        if (i % 4 == 0) {noisy.to[i] += cv::Point2f(35, -25);}
      }
      const auto robust = auv_flow::estimateMetric(noisy.from, noisy.to, .1, a, b, 2, c, settings);
      const auto error = robust.velocity_body-expected_body;
      squared_error += error.dot(error);
      if (error.dot(robust.covariance_body.inv()*error) < 7.815) {++covered;}
      require(robust.inliers.size() >= tracks.from.size()*0.7 &&
        robust.inliers.size() < tracks.from.size()*0.8, "Reject 25 percent moving-object/outlier tracks");
    }
    const double rmse = std::sqrt(squared_error/30);
    std::cout << "Noisy/outlier velocity RMSE: " << rmse << " m/s; covariance coverage " << covered << "/30\n";
    require(rmse < .02, "Noisy metric velocity RMSE must be below 0.02 m/s");
    require(covered >= 27, "Conservative covariance must cover at least 90 percent of noisy trials");

    rejected([&]() {auv_flow::estimateMetric({}, {}, .1, a, b, 2, c, settings);}, "no features");
    rejected([&]() {auv_flow::estimateMetric(tracks.from, tracks.to, .1, a, b, 0, c, settings);}, "unknown height");
    auto unknown = c;
    unknown.calibrated = false;
    rejected([&]() {auv_flow::estimateMetric(tracks.from, tracks.to, .1, a, b, 2, unknown, settings);}, "unknown calibration");
    unknown = c;
    unknown.extrinsics_calibrated = false;
    rejected([&]() {unknown.validate();}, "unknown mounting");
    auto zero_cov = settings;
    zero_cov.height_std_m = 0;
    rejected([&]() {auv_flow::estimateMetric(tracks.from, tracks.to, .1, a, b, 2, c, zero_cov);}, "unknown height uncertainty");
    rejected([&]() {auv_flow::estimateMetric(tracks.from, tracks.to, 0, a, b, 2, c, settings);}, "duplicate timestamps");
    auto invalid_orientation = b;
    invalid_orientation.q = {0, 0, 0, 0};
    rejected([&]() {auv_flow::estimateMetric(tracks.from, tracks.to, .1, a, invalid_orientation, 2, c, settings);}, "invalid quaternion");
    auto bad_tracks = tracks;
    bad_tracks.to[0].x = std::numeric_limits<float>::quiet_NaN();
    rejected([&]() {auv_flow::estimateMetric(bad_tracks.from, bad_tracks.to, .1, a, b, 2, c, settings);}, "nonfinite pixels");
    bad_tracks = tracks;
    for (size_t i = 0; i < bad_tracks.to.size(); ++i) {
      bad_tracks.to[i] = cv::Point2f(60+(i*83)%500, 60+(i*67)%350);
    }
    rejected([&]() {auv_flow::estimateMetric(bad_tracks.from, bad_tracks.to, .1, a, b, 2, c, settings);}, "incoherent motion");
    auto upward = c;
    upward.r_bc = cv::Matx33d::eye();
    rejected([&]() {auv_flow::estimateMetric(tracks.from, tracks.to, .1, a, b, 2, upward, settings);}, "upward camera");
    bad_tracks = tracks;
    for (size_t i = 0; i < bad_tracks.from.size(); ++i) {
      bad_tracks.from[i] = {static_cast<float>(80+i*4), 240};
      bad_tracks.to[i] = bad_tracks.from[i];
    }
    const auv_flow::Orientation level0{1000000000LL, {0, 0, 0, 1}, 1e-8};
    const auv_flow::Orientation level1{1100000000LL, {0, 0, 0, 1}, 1e-8};
    rejected([&]() {auv_flow::estimateMetric(bad_tracks.from, bad_tracks.to, .1, level0, level1, 2, c, settings);}, "collinear features");

    auv_flow::OrientationBuffer buffer(.11);
    buffer.add(a);
    buffer.add(b);
    const auto mid = buffer.sample(1050000000LL);
    require(cv::norm(mid.q-auv_flow::slerp(a.q, b.q, .5)) < 1e-10, "Timestamp interpolation");
    rejected([&]() {buffer.sample(1200000000LL);}, "orientation extrapolation");
    auv_flow::OrientationBuffer stale(.05);
    stale.add(a);
    stale.add(b);
    rejected([&]() {stale.sample(1050000000LL);}, "stale orientation bracket");
    auto no_variance = a;
    no_variance.variance = 0;
    rejected([&]() {buffer.add(no_variance);}, "unknown orientation covariance");
    auto almost_unit = a;
    almost_unit.q *= 0.995;
    buffer.add(almost_unit);
    require(std::abs(cv::norm(buffer.sample(a.stamp_ns).q)-1.0) < 1e-12,
      "Exact-time orientation must also be normalized");

    cv::Mat image(c.size, CV_8UC1, cv::Scalar(0));
    cv::RNG rng(42);
    for (int i = 0; i < 180; ++i) {
      cv::rectangle(image, cv::Rect(rng.uniform(30, 600), rng.uniform(30, 440), 6, 6),
        cv::Scalar(255), -1);
    }
    auv_flow::FlowTracker tracker;
    require(!tracker.update(image).valid, "Image-to-metric pipeline first frame has no velocity");
    cv::Mat shifted;
    const cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, 4, 0, 1, -3);
    cv::warpAffine(image, shifted, shift, image.size());
    const auto flow = tracker.update(shifted);
    const auto image_estimate = auv_flow::estimateMetric(flow.from, flow.to, .1,
      level0, level1, 2, c, settings);
    require(cv::norm(image_estimate.velocity_body-cv::Vec3d(-.19, -3*1.9/(410*.1), 0)) < .003,
      "Actual Shi-Tomasi/LK tracks must convert to calibrated body velocity in m/s");
    if (argc > 1) {
      auto saved = auv_flow::BottomCalibration::load(argv[1]);
      require(saved.size == cv::Size(640, 360), "Saved calibration dimensions");
      // The user's configuration is editable; test rejection explicitly rather
      // than relying on their current confirmation flags remaining zero.
      saved.calibrated = false;
      rejected([&]() {saved.validate();}, "unconfirmed saved calibration");
    }
    std::cout << checks << " quantitative/gating checks passed\n";
  } catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
