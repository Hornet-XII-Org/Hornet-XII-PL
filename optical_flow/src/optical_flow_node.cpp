#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include "optical_flow/flow_tracker.hpp"
#include "optical_flow/image_message.hpp"
#include "optical_flow/metric_estimator.hpp"

class OpticalFlowNode : public rclcpp::Node
{
  using Clock = std::chrono::steady_clock;
  using Image = sensor_msgs::msg::Image;
  struct Pending {Image::SharedPtr image; Clock::time_point received;};
public:
  OpticalFlowNode() : Node("optical_flow_node")
  {
    reference_frame_ = declare_parameter<std::string>("reference_frame", "odom");
    child_frame_ = declare_parameter<std::string>("child_frame", "base_link");
    assume_fixed_orientation_ = declare_parameter<bool>("assume_fixed_orientation", false);
    fixed_orientation_std_rad_ = declare_parameter<double>(
      "fixed_orientation_std_rad", 0.017453292519943295);
    auv_flow::assumedLevelOrientation(1, fixed_orientation_std_rad_);
    if (assume_fixed_orientation_) {
      RCLCPP_WARN(get_logger(),
        "TEST MODE: assuming level body, zero yaw and no rotation at every image timestamp; "
        "velocities depend on this assumption and the configured camera model");
    }
    height_ = declare_parameter<double>("body_height_m", 0.0);
    settings_.height_std_m = declare_parameter<double>("height_std_m", 0.0);
    settings_.min_tracks = declare_parameter<int>("min_tracks", 12);
    settings_.min_inlier_ratio = declare_parameter<double>("min_inlier_ratio", 0.6);
    settings_.residual_threshold_m = declare_parameter<double>("residual_threshold_m", 0.025);
    settings_.pixel_noise_px = declare_parameter<double>("pixel_noise_px", 0.5);
    settings_.velocity_noise_mps = declare_parameter<double>("velocity_noise_mps", 0.02);
    settings_.focal_relative_std = declare_parameter<double>("focal_relative_std", 0.01);
    settings_.principal_point_std_px = declare_parameter<double>("principal_point_std_px", 0.5);
    settings_.extrinsic_angle_std_rad = declare_parameter<double>("extrinsic_angle_std_rad", 0.01);
    settings_.lever_arm_std_m = declare_parameter<double>("lever_arm_std_m", 0.005);
    settings_.unobserved_variance = declare_parameter<double>("unobserved_variance", 1e6);
    settings_.max_speed_mps = declare_parameter<double>("max_speed_mps", 5.0);
    settings_.min_ray_cosine = declare_parameter<double>("min_ray_cosine", 0.2);
    min_dt_ = declare_parameter<double>("min_dt_s", 0.005);
    max_dt_ = declare_parameter<double>("max_dt_s", 0.2);
    wait_ = declare_parameter<double>("orientation_wait_s", 0.15);
    const double orientation_gap = declare_parameter<double>("orientation_max_gap_s", 0.1);
    const int capacity = declare_parameter<int>("max_pending_images", 5);
    const int threads = declare_parameter<int>("opencv_threads", 0);
    if (reference_frame_.empty() || child_frame_.empty() || reference_frame_ == child_frame_ ||
      !std::isfinite(height_) || height_ < 0 || !std::isfinite(min_dt_) ||
      !std::isfinite(max_dt_) || !(0 < min_dt_ && min_dt_ < max_dt_) ||
      !std::isfinite(wait_) || wait_ <= 0 || capacity <= 0 || threads < 0)
    {
      throw std::runtime_error("Invalid frames, scale, timing, queue or threading configuration");
    }
    if (threads > 0) {cv::setNumThreads(threads);}
    capacity_ = static_cast<size_t>(capacity);
    orientations_ = std::make_unique<auv_flow::OrientationBuffer>(orientation_gap);
    const auto calibration_path = declare_parameter<std::string>("calibration_file",
      ament_index_cpp::get_package_share_directory("optical_flow") + "/config/bottom_camera.yaml");
    calibration_ = auv_flow::BottomCalibration::load(calibration_path);
    // A template/unknown scale is permitted at startup, but never produces odometry.
    try {
      calibration_.validate();
      settings_.validate();
      if (height_ <= 0 || calibration_.body_frame != child_frame_) {
        throw std::runtime_error("Body height unknown or calibration body frame mismatches child_frame");
      }
      ready_ = true;
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "Metric output disabled: %s", e.what());
    }
    odometry_ = create_publisher<nav_msgs::msg::Odometry>("/bottom/flow/odometry", 10);
    detected_ = create_publisher<Image>("/bottom/flow/image", rclcpp::QoS(5).reliable());
    valid_ = create_publisher<std_msgs::msg::Bool>("/bottom/flow/valid", 10);
    assumption_status_ = create_publisher<std_msgs::msg::Bool>(
      "/bottom/flow/assumed_orientation", rclcpp::QoS(1).reliable().transient_local());
    std_msgs::msg::Bool assumption;
    assumption.data = assume_fixed_orientation_;
    assumption_status_->publish(assumption);
    auto input_qos = rclcpp::QoS(rclcpp::SensorDataQoS());
    if (declare_parameter<bool>("input_reliable", false)) {input_qos.reliable();}
    image_sub_ = create_subscription<Image>("/bottom/image_rect", input_qos,
      [this](Image::SharedPtr image) {
        if (pending_.size() >= capacity_) {
          const auto expired = pending_.front().image;
          pending_.pop_front();
          reset();
          process(expired, nullptr);
        }
        last_input_ = Clock::now();
        pending_.push_back({image, last_input_});
        drain();
      });
    orientation_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/bottom/orientation", rclcpp::SensorDataQoS(),
      [this](geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
        if (assume_fixed_orientation_) {return;}
        try {
          if (msg->header.frame_id != reference_frame_) {
            throw std::runtime_error("Orientation reference frame mismatch");
          }
          cv::Matx33d covariance;
          for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
              covariance(row, col) = msg->pose.covariance[(row+3)*6+col+3];
            }
          }
          cv::Vec3d eigenvalues;
          if (!auv_flow::finite(covariance) ||
            cv::norm(covariance-covariance.t()) > 1e-8)
          {
            throw std::runtime_error("Invalid orientation covariance");
          }
          cv::eigen(covariance, eigenvalues);
          if (eigenvalues[2] <= 0) {
            throw std::runtime_error("Orientation covariance is unknown or nonpositive");
          }
          const auto & q = msg->pose.pose.orientation;
          orientations_->add({stampNs(msg->header.stamp), {q.x, q.y, q.z, q.w}, eigenvalues[0]});
          drain();
        } catch (const std::exception & e) {warn(e.what());}
      });
    timer_ = create_wall_timer(std::chrono::milliseconds(10), [this]() {
        drain();
        if (Clock::now()-last_input_ > std::chrono::milliseconds(500) && metric_valid_) {
          reset();
          publishValidity(false);
        }
      });
    stats_timer_ = create_wall_timer(std::chrono::seconds(2), [this]() {
        const auto time = Clock::now();
        const double elapsed = std::chrono::duration<double>(time-stats_start_).count();
        RCLCPP_INFO(get_logger(), "Flow: %.1f images/s | %.1f metric odometry/s | %.1f ms/image",
          frames_/elapsed, outputs_/elapsed, frames_ ? total_ms_/frames_ : 0.0);
        stats_start_ = time;
        frames_ = outputs_ = 0;
        total_ms_ = 0;
      });
  }

private:
  static int64_t stampNs(const builtin_interfaces::msg::Time & stamp)
  {
    if (stamp.sec < 0 || stamp.nanosec >= 1000000000U) {
      throw std::runtime_error("Invalid acquisition timestamp");
    }
    return static_cast<int64_t>(stamp.sec)*1000000000LL + stamp.nanosec;
  }
  void reset()
  {
    tracker_.reset();
    previous_stamp_ = 0;
    previous_orientation_valid_ = false;
  }
  void warn(const std::string & reason)
  {
    if (Clock::now()-last_warning_ > std::chrono::seconds(2)) {
      RCLCPP_WARN(get_logger(), "Odometry withheld: %s", reason.c_str());
      last_warning_ = Clock::now();
    }
  }
  void publishValidity(bool value)
  {
    metric_valid_ = value;
    std_msgs::msg::Bool msg;
    msg.data = value;
    valid_->publish(msg);
  }
  void drain()
  {
    while (!pending_.empty()) {
      auv_flow::Orientation orientation;
      if (ready_) {
        try {
          const auto stamp = stampNs(pending_.front().image->header.stamp);
          orientation = assume_fixed_orientation_ ?
            auv_flow::assumedLevelOrientation(stamp, fixed_orientation_std_rad_) :
            orientations_->sample(stamp);
        }
        catch (const std::exception & e) {
          if (std::chrono::duration<double>(Clock::now()-pending_.front().received).count() < wait_) {
            return;
          }
          warn(e.what());
          const auto image = pending_.front().image;
          pending_.pop_front();
          // Missing attitude invalidates metric motion, not the visual preview.
          process(image, nullptr);
          continue;
        }
      }
      const auto image = pending_.front().image;
      pending_.pop_front();
      process(image, ready_ ? &orientation : nullptr);
    }
  }
  void process(const Image::SharedPtr & image, const auv_flow::Orientation * orientation)
  {
    const auto started = Clock::now();
    bool valid = false;
    try {
      uint64_t raw_stamp;
      cv::Mat frame = auv_flow::unpackImage(*image, raw_stamp);
      const auto stamp = static_cast<int64_t>(raw_stamp);
      if (stamp <= 0 || frame.size() != calibration_.size ||
        image->header.frame_id != calibration_.camera_frame)
      {
        throw std::runtime_error("Image timestamp, dimensions or optical frame mismatches calibration");
      }
      const double dt = (stamp-previous_stamp_)*1e-9;
      if (previous_stamp_ && (dt < min_dt_ || dt > max_dt_)) {reset();}
      cv::Mat gray;
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      const auto tracks = tracker_.update(gray);
      std::string text = "Metric velocity unavailable";
      if (ready_ && !orientation) {text = "No orientation - tracking preview only";}
      if (ready_ && previous_stamp_ && previous_orientation_valid_ && orientation) {
        try {
          const auto estimate = auv_flow::estimateMetric(tracks.from, tracks.to,
            dt, previous_orientation_, *orientation, height_, calibration_, settings_);
          nav_msgs::msg::Odometry odometry;
          odometry.header.stamp = image->header.stamp;
          odometry.header.frame_id = reference_frame_;
          odometry.child_frame_id = child_frame_;
          odometry.pose.pose.orientation.x = orientation->q[0];
          odometry.pose.pose.orientation.y = orientation->q[1];
          odometry.pose.pose.orientation.z = orientation->q[2];
          odometry.pose.pose.orientation.w = orientation->q[3];
          for (int axis = 0; axis < 6; ++axis) {
            odometry.pose.covariance[axis*6+axis] = settings_.unobserved_variance;
            odometry.twist.covariance[axis*6+axis] = settings_.unobserved_variance;
          }
          const auto & v = estimate.velocity_body;
          odometry.twist.twist.linear.x = v[0];
          odometry.twist.twist.linear.y = v[1];
          odometry.twist.twist.linear.z = v[2];
          for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
              odometry.twist.covariance[row*6+col] = estimate.covariance_body(row, col);
            }
          }
          odometry_->publish(odometry);
          ++outputs_;
          valid = true;
          text = cv::format("%s: vx %.3f vy %.3f m/s", child_frame_.c_str(), v[0], v[1]);
          if (assume_fixed_orientation_) {text = "TEST fixed attitude | " + text;}
          for (const auto i : estimate.inliers) {
            cv::arrowedLine(frame, tracks.from[i], tracks.to[i],
              cv::Scalar(0, 255, 0), 1);
          }
        } catch (const std::exception & e) {warn(e.what());}
      }
      previous_stamp_ = stamp;
      previous_orientation_valid_ = orientation != nullptr;
      if (orientation) {previous_orientation_ = *orientation;}
      for (const auto & p : tracks.corners) {cv::circle(frame, p, 2, cv::Scalar(0, 255, 255), -1);}
      cv::putText(frame, text, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(0, 0, 0), 3);
      cv::putText(frame, text, {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
        cv::Scalar(255, 255, 255), 1);
      auto annotated = auv_flow::packImage(frame, raw_stamp);
      annotated.header = image->header;
      detected_->publish(annotated);
    } catch (const std::exception & e) {reset(); warn(e.what());}
    publishValidity(valid);
    ++frames_;
    total_ms_ += std::chrono::duration<double, std::milli>(Clock::now()-started).count();
  }

  auv_flow::BottomCalibration calibration_;
  auv_flow::MetricSettings settings_;
  auv_flow::FlowTracker tracker_;
  std::unique_ptr<auv_flow::OrientationBuffer> orientations_;
  auv_flow::Orientation previous_orientation_;
  std::deque<Pending> pending_;
  std::string reference_frame_, child_frame_;
  double height_, min_dt_, max_dt_, wait_, fixed_orientation_std_rad_;
  size_t capacity_;
  bool ready_{false}, metric_valid_{false}, previous_orientation_valid_{false};
  bool assume_fixed_orientation_{false};
  int64_t previous_stamp_{0};
  Clock::time_point last_input_{Clock::now()}, last_warning_{Clock::now()-std::chrono::seconds(3)};
  Clock::time_point stats_start_{Clock::now()};
  size_t frames_{0}, outputs_{0};
  double total_ms_{0};
  rclcpp::Subscription<Image>::SharedPtr image_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr orientation_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_;
  rclcpp::Publisher<Image>::SharedPtr detected_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr valid_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr assumption_status_;
  rclcpp::TimerBase::SharedPtr timer_, stats_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {rclcpp::spin(std::make_shared<OpticalFlowNode>());}
  catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("optical_flow_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
