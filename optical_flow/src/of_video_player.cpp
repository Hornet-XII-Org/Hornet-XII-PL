#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include "optical_flow/image_message.hpp"

// Change this absolute path to select the default input video, then rebuild.
const std::string VIDEO_PATH = "/home/tunglam/Hornet_OF/optical_flow/wiggle.MOV";

class VideoNode : public rclcpp::Node
{
public:
  VideoNode() : Node("of_video_player")
  {
    const int threads = declare_parameter<int>("opencv_threads", 0);
    if (threads < 0) {
      throw std::runtime_error("opencv_threads must be nonnegative (0 keeps OpenCV defaults)");
    }
    if (threads > 0) {
      cv::setNumThreads(threads);
    }
    RCLCPP_INFO(get_logger(), "OpenCV threads: %d", cv::getNumThreads());
    stats_timer_ = create_wall_timer(std::chrono::seconds(2), [this]() {reportStats();});
    const auto override_path = declare_parameter<std::string>("video_path", "");
    const auto path = override_path.empty() ? VIDEO_PATH : override_path;
    loop_ = declare_parameter<bool>("loop", false);
    image_width_ = declare_parameter<int>("image_width", 640);
    frame_id_ = declare_parameter<std::string>("frame_id", "bottom_camera_optical_frame");
    if (image_width_ < 0) {
      throw std::runtime_error("image_width must be nonnegative (0 keeps source resolution)");
    }
    const double requested_fps = declare_parameter<double>("fps", 0.0);
    if (!std::isfinite(requested_fps) || requested_fps < 0) {
      throw std::runtime_error("fps must be zero (source rate) or positive");
    }
    if (!capture_.open(path)) {
      throw std::runtime_error("Cannot open video: " + path);
    }
    fps_ = requested_fps > 0 ? requested_fps : capture_.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(fps_) || fps_ <= 0) {
      throw std::runtime_error("Video has no valid frame rate; set fps");
    }
    const auto period = std::chrono::nanoseconds(static_cast<int64_t>(1e9 / fps_));
    if (period.count() <= 0) {
      throw std::runtime_error("fps is too large");
    }
    publisher_ = create_publisher<sensor_msgs::msg::Image>(
      "video_input", rclcpp::QoS(rclcpp::KeepLast(5)).reliable().durability_volatile());
    epoch_ = now().nanoseconds();
    timer_ = create_wall_timer(period, [this]() {publishFrame();});
    RCLCPP_INFO(get_logger(), "Playing %s at %.3f FPS", path.c_str(), fps_);
  }

private:
  void publishFrame()
  {
    const auto started = Clock::now();
    cv::Mat frame;
    if (!capture_.read(frame)) {
      if (!loop_ || !capture_.set(cv::CAP_PROP_POS_FRAMES, 0) || !capture_.read(frame)) {
        timer_->cancel();
        RCLCPP_INFO(get_logger(), "Video playback finished");
        return;
      }
      // Force a tracker reset across the discontinuity at the loop boundary.
      frame_index_ += static_cast<uint64_t>(std::ceil(fps_));
    }
    const auto stamp = epoch_ + static_cast<int64_t>(std::llround(frame_index_ * 1e9 / fps_));
    if (image_width_ > 0 && frame.cols > image_width_) {
      const int height = std::max(1, static_cast<int>(std::lround(
        static_cast<double>(frame.rows) * image_width_ / frame.cols)));
      cv::resize(frame, frame, cv::Size(image_width_, height), 0, 0, cv::INTER_AREA);
    }
    const auto work_done = Clock::now();
    publisher_->publish(auv_flow::packImage(frame, static_cast<uint64_t>(stamp), frame_id_));
    ++frame_index_;
    ++stats_frames_;
    stats_width_ = frame.cols;
    stats_height_ = frame.rows;
    stats_work_ms_ += std::chrono::duration<double, std::milli>(work_done - started).count();
    stats_total_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  }
  void reportStats()
  {
    const auto time = Clock::now();
    const double elapsed = std::chrono::duration<double>(time - stats_start_).count();
    const double count = static_cast<double>(std::max<size_t>(1, stats_frames_));
    RCLCPP_INFO(get_logger(),
      "Playback: %.1f publish calls/s | callback %.1f ms | decode+resize %.1f ms | %dx%d",
      stats_frames_ / elapsed, stats_total_ms_ / count, stats_work_ms_ / count,
      stats_width_, stats_height_);
    stats_start_ = time;
    stats_frames_ = 0;
    stats_total_ms_ = stats_work_ms_ = 0;
  }
  using Clock = std::chrono::steady_clock;
  Clock::time_point stats_start_{Clock::now()};
  size_t stats_frames_{0};
  double stats_total_ms_{0}, stats_work_ms_{0};
  int stats_width_{0}, stats_height_{0};
  rclcpp::TimerBase::SharedPtr stats_timer_;
  cv::VideoCapture capture_;
  bool loop_;
  int image_width_;
  std::string frame_id_;
  double fps_;
  int64_t epoch_;
  uint64_t frame_index_{0};
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<VideoNode>());
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("of_video_player"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
