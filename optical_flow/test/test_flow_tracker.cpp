#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <opencv2/videoio.hpp>
#include "optical_flow/flow_tracker.hpp"

void require(bool ok, const char * message)
{
  if (!ok) {
    throw std::runtime_error(message);
  }
}

int main(int argc, char ** argv)
{
  try {
    cv::Mat image(480, 640, CV_8UC1, cv::Scalar(0));
    cv::RNG random(42);
    for (int i = 0; i < 200; ++i) {
      const cv::Point p(random.uniform(20, 620), random.uniform(20, 460));
      cv::rectangle(image, cv::Rect(p.x, p.y, 7, 7), cv::Scalar(255), -1);
    }
    auv_flow::FlowTracker tracker;
    require(!tracker.update(image).valid, "First frame should only initialize features");
    cv::Mat shifted;
    const cv::Mat translation = (cv::Mat_<double>(2, 3) << 1, 0, 4, 0, 1, -3);
    cv::warpAffine(image, shifted, translation, image.size());
    const auto result = tracker.update(shifted);
    require(result.valid, "Known translation should track");
    require(cv::norm(result.displacement - cv::Point2d(4, -3)) < 0.2,
      "Translation estimate is inaccurate");
    const auto stationary = tracker.update(shifted);
    require(stationary.valid && cv::norm(stationary.displacement) < 0.1,
      "Stationary image should give zero velocity");
    cv::Mat large, large_shifted;
    cv::resize(image, large, cv::Size(1280, 960));
    const cv::Mat large_translation = (cv::Mat_<double>(2, 3) << 1, 0, 8, 0, 1, -6);
    cv::warpAffine(large, large_shifted, large_translation, large.size());
    auv_flow::FlowTracker scaled_tracker;
    scaled_tracker.update(large);
    const auto scaled = scaled_tracker.update(large_shifted);
    require(scaled.valid && cv::norm(scaled.displacement - cv::Point2d(8, -6)) < 0.4,
      "Internal downsampling must preserve input-pixel displacement units");
    tracker.reset();
    require(!tracker.update(shifted).valid, "Reset must prevent cross-gap motion");
    tracker.update(cv::Mat::zeros(image.size(), CV_8UC1));
    require(!tracker.update(cv::Mat::zeros(image.size(), CV_8UC1)).valid,
      "Blank images must not produce velocity");
    require(!tracker.update(image).valid && tracker.update(shifted).valid,
      "Tracker must recover when features return");
    require(!tracker.update(cv::Mat::zeros(240, 320, CV_8UC1)).valid,
      "Resolution changes must reinitialize tracking");
    if (argc > 1) {
      cv::VideoCapture video(argv[1]);
      require(video.isOpened(), "Cannot open test video");
      tracker.reset();
      cv::Mat frame, gray;
      int frames = 0, valid = 0;
      const auto start = std::chrono::steady_clock::now();
      while (video.read(frame)) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        const auto flow = tracker.update(gray);
        valid += flow.valid ? 1 : 0;
        ++frames;
      }
      const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
      std::cout << "Decode + tracking: " << frames / seconds << " FPS\n";
      std::cout << "Video: " << valid << " valid motion estimates / " << frames << " frames\n";
      require(frames > 1 && valid > (frames - 1) * 0.8, "Insufficient video tracking coverage");
    }
    std::cout << "Flow tests passed\n";
  } catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
