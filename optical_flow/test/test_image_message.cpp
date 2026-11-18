#include <iostream>
#include <stdexcept>
#include "optical_flow/image_message.hpp"

int main()
{
  try {
    cv::Mat source(10, 20, CV_8UC3);
    cv::randu(source, 0, 255);
    const cv::Mat roi = source(cv::Rect(2, 2, 8, 5));
    constexpr uint64_t expected_stamp = 1234567890123456789ULL;
    auto msg = auv_flow::packImage(roi, expected_stamp);
    uint64_t stamp = 0;
    const auto decoded = auv_flow::unpackImage(msg, stamp);
    if (stamp != expected_stamp || cv::norm(roi, decoded, cv::NORM_INF) != 0 ||
      msg.encoding != "bgr8" || msg.header.frame_id != "bottom_camera_optical_frame")
    {
      throw std::runtime_error("Image/timestamp round trip failed");
    }
    // Standard Image publishers may pad rows; the decoder must respect step.
    auto padded = msg;
    padded.step += 5;
    padded.data.assign(padded.height * padded.step, 0);
    for (uint32_t row = 0; row < msg.height; ++row) {
      std::memcpy(padded.data.data() + row * padded.step,
        msg.data.data() + row * msg.step, msg.step);
    }
    if (cv::norm(roi, auv_flow::unpackImage(padded, stamp), cv::NORM_INF) != 0) {
      throw std::runtime_error("Padded rows decoded incorrectly");
    }
    msg.data.pop_back();
    bool rejected = false;
    try {
      auv_flow::unpackImage(msg, stamp);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    if (!rejected) {
      throw std::runtime_error("Truncated image was accepted");
    }
    sensor_msgs::msg::Image mono;
    mono.header = padded.header;
    mono.height = 2;
    mono.width = 2;
    mono.encoding = "mono8";
    mono.step = 3;
    mono.data = {10, 20, 99, 30, 40, 99};
    const auto mono_bgr = auv_flow::unpackImage(mono, stamp);
    if (mono_bgr.at<cv::Vec3b>(1, 0) != cv::Vec3b(30, 30, 30)) {
      throw std::runtime_error("Padded mono8 conversion failed");
    }
    auto rgb = auv_flow::packImage(cv::Mat(1, 1, CV_8UC3, cv::Scalar(10, 20, 30)), expected_stamp);
    rgb.encoding = "rgb8";
    if (auv_flow::unpackImage(rgb, stamp).at<cv::Vec3b>(0, 0) != cv::Vec3b(30, 20, 10)) {
      throw std::runtime_error("RGB channel conversion failed");
    }
    std::cout << "Image message tests passed\n";
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
