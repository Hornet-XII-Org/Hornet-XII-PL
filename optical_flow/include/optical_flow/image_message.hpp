#pragma once
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace auv_flow
{
inline sensor_msgs::msg::Image packImage(
  const cv::Mat & image, uint64_t stamp, const std::string & frame_id = "bottom_camera_optical_frame")
{
  if (image.empty() || image.type() != CV_8UC3 ||
    image.total() > std::numeric_limits<uint32_t>::max() / 3 ||
    stamp / 1000000000ULL > INT32_MAX)
  {
    throw std::runtime_error("Expected a BGR8 image and a supported ROS timestamp");
  }
  sensor_msgs::msg::Image msg;
  msg.header.stamp.sec = static_cast<int32_t>(stamp / 1000000000ULL);
  msg.header.stamp.nanosec = static_cast<uint32_t>(stamp % 1000000000ULL);
  msg.header.frame_id = frame_id;
  msg.height = image.rows;
  msg.width = image.cols;
  msg.encoding = "bgr8";
  msg.is_bigendian = false;
  msg.step = image.cols * 3;
  msg.data.resize(image.total() * 3);
  for (int row = 0; row < image.rows; ++row) {
    std::memcpy(msg.data.data() + static_cast<size_t>(row) * msg.step,
      image.ptr(row), msg.step);
  }
  return msg;
}

inline cv::Mat unpackImage(const sensor_msgs::msg::Image & msg, uint64_t & stamp)
{
  const bool mono = msg.encoding == "mono8";
  const bool rgb = msg.encoding == "rgb8";
  const uint64_t channels = mono ? 1 : 3;
  if ((!mono && !rgb && msg.encoding != "bgr8") || msg.height == 0 || msg.width == 0 ||
    msg.height > INT32_MAX || msg.width > INT32_MAX ||
    msg.step < static_cast<uint64_t>(msg.width) * channels ||
    msg.data.size() != static_cast<uint64_t>(msg.height) * msg.step ||
    msg.header.stamp.sec < 0 || msg.header.stamp.nanosec >= 1000000000U)
  {
    throw std::runtime_error("Invalid mono8/rgb8/bgr8 Image encoding, dimensions, stride or timestamp");
  }
  stamp = static_cast<uint64_t>(msg.header.stamp.sec) * 1000000000ULL +
    msg.header.stamp.nanosec;
  const cv::Mat input(msg.height, msg.width, mono ? CV_8UC1 : CV_8UC3,
    const_cast<uint8_t *>(msg.data.data()), msg.step);
  if (mono || rgb) {
    cv::Mat bgr;
    cv::cvtColor(input, bgr, mono ? cv::COLOR_GRAY2BGR : cv::COLOR_RGB2BGR);
    return bgr;
  }
  return input.clone();
}
}  // namespace auv_flow
