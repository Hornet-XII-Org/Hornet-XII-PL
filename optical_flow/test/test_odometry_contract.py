#!/usr/bin/env python3
"""Run the actual C++ node: metric units/headers/covariance and invalid withholding."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

# Isolate the fixture from a running AUV or developer's graph.
os.environ['ROS_DOMAIN_ID'] = str(20 + os.getpid() % 70)
os.environ['ROS_LOCALHOST_ONLY'] = '1'
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def wait_until(node, predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.01)
        if predicate():
            return
    raise AssertionError('Timed out waiting for node/discovery/output')


def spin_for(node, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.01)


def main():
    binary = sys.argv[1]
    rclpy.init()
    node = rclpy.create_node('flow_contract_driver')
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    images = node.create_publisher(Image, '/bottom/image_rect', qos)
    orientations = node.create_publisher(PoseWithCovarianceStamped, '/bottom/orientation', qos)
    outputs = []
    sub = node.create_subscription(Odometry, '/bottom/flow/odometry', outputs.append, qos)
    previews = []
    preview_sub = node.create_subscription(Image, '/bottom/flow/image', previews.append, qos)
    random = np.random.default_rng(42)
    image = np.zeros((360, 640), np.uint8)
    for x, y in random.integers([25, 25], [610, 330], size=(180, 2)):
        cv2.rectangle(image, (int(x), int(y)), (int(x+6), int(y+6)), 255, -1)
    with tempfile.TemporaryDirectory(prefix='flow_contract_') as directory:
        path = Path(directory) / 'calibration.yaml'
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
        for key, value in {'calibrated': 1, 'extrinsics_calibrated': 1,
                           'image_width': 640, 'image_height': 360,
                           'camera_frame': 'bottom_camera_optical_frame',
                           'body_frame': 'base_link'}.items():
            storage.write(key, value)
        storage.write('rectified_camera_matrix', np.array(
            [[400., 0., 319.5], [0., 400., 179.5], [0., 0., 1.]]))
        storage.write('camera_to_body_quaternion_xyzw', np.array([[1.], [0.], [0.], [0.]]))
        storage.write('camera_position_body_m', np.zeros((3, 1)))
        storage.release()

        def send(index, pixels, with_orientation=True, offset=0):
            nanoseconds = 10_000_000_000 + index * 100_000_000 + offset
            msg = Image()
            msg.header.stamp.sec = nanoseconds // 1_000_000_000
            msg.header.stamp.nanosec = nanoseconds % 1_000_000_000
            msg.header.frame_id = 'bottom_camera_optical_frame'
            if with_orientation:
                pose = PoseWithCovarianceStamped()
                pose.header.stamp = msg.header.stamp
                pose.header.frame_id = 'odom'
                pose.pose.pose.orientation.w = 1.0
                for axis in range(3, 6):
                    pose.pose.covariance[axis*6+axis] = 1e-8
                orientations.publish(pose)
                spin_for(node, .025)
            msg.encoding = 'mono8'
            msg.height, msg.width = pixels.shape
            msg.step = msg.width
            msg.data = pixels.tobytes()
            images.publish(msg)
            spin_for(node, .12)
            return nanoseconds

        for height, assume_fixed in ((0.0, False), (2.0, False), (0.0, True), (2.0, True)):
            outputs.clear()
            previews.clear()
            log = open(Path(directory) / f'node_{height}_{assume_fixed}.log', 'w+')
            process = subprocess.Popen([
                binary, '--ros-args', '-p', f'calibration_file:={path}',
                '-p', f'body_height_m:={height}', '-p', 'height_std_m:=0.01',
                '-p', f'assume_fixed_orientation:={str(assume_fixed).lower()}',
                '-p', 'input_reliable:=true', '-p', 'opencv_threads:=1'],
                stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_until(node, lambda: process.poll() is not None or (
                    images.get_subscription_count() > 0 and orientations.get_subscription_count() > 0
                    and node.count_publishers('/bottom/flow/odometry') > 0
                    and node.count_publishers('/bottom/flow/image') > 0))
                assert process.poll() is None, 'C++ node failed at startup'
                # Reproduce the black preview: calibrated node, no attitude samples at all.
                preview_stamp = send(-2, image, with_orientation=False)
                wait_until(node, lambda: any(
                    p.header.stamp.sec * 1_000_000_000 + p.header.stamp.nanosec == preview_stamp
                    for p in previews))
                preview = next(p for p in previews if (
                    p.header.stamp.sec * 1_000_000_000 + p.header.stamp.nanosec == preview_stamp))
                assert preview.encoding == 'bgr8' and preview.width == 640 and preview.height == 360
                assert preview.header.frame_id == 'bottom_camera_optical_frame'
                assert max(preview.data) > 0, 'Missing orientation produced a blank preview'
                assert not outputs, 'Missing startup orientation produced odometry'
                shifted = cv2.warpAffine(image, np.array([[1., 0., -2.], [0., 1., 1.]]), (640, 360))
                send(0, image, with_orientation=not assume_fixed)
                stamp = send(1, shifted, with_orientation=not assume_fixed)
                if height == 0.0:
                    spin_for(node, .25)
                    assert not outputs, 'Unknown scale published a fabricated measurement'
                    continue
                wait_until(node, lambda: any(
                    o.header.stamp.sec * 1_000_000_000 + o.header.stamp.nanosec == stamp
                    for o in outputs))
                result = next(o for o in outputs if (
                    o.header.stamp.sec * 1_000_000_000 + o.header.stamp.nanosec == stamp))
                assert result.header.frame_id == 'odom' and result.child_frame_id == 'base_link'
                assert result.header.stamp.sec * 1_000_000_000 + result.header.stamp.nanosec == stamp
                velocity = result.twist.twist.linear
                # Optical right maps to body X; optical down maps to body -Y.
                assert abs(velocity.x - .1) < .015 and abs(velocity.y - .05) < .015, (
                    velocity.x, velocity.y)
                covariance = np.array(result.twist.covariance).reshape(6, 6)
                assert np.linalg.eigvalsh(covariance[:3, :3]).min() > 0
                assert covariance[2, 2] >= 1e5 and min(covariance.diagonal()[3:]) >= 1e5
                assert min(np.array(result.pose.covariance).reshape(6, 6).diagonal()) >= 1e5
                if assume_fixed:
                    count = len(outputs)
                    send(2, np.zeros_like(image), with_orientation=False)
                    send(3, np.zeros_like(image), with_orientation=False)
                    assert len(outputs) == count, 'Fixed test mode bypassed feature validation'
                    print('Fixed-orientation test passed: metric output without attitude messages; '
                          'unknown height and feature loss still withheld.')
                    continue
                # No sufficiently recent orientation: no odometry, even though images arrive.
                count = len(outputs)
                send(2, image, with_orientation=False)
                spin_for(node, .3)
                assert len(outputs) == count, 'Stale/missing orientation produced odometry'
                # Metric estimation must not span an image with missing attitude,
                # even when consecutive image timestamps are within max_dt_s.
                send(3, image)
                assert len(outputs) == count
                send(4, np.zeros_like(image))
                send(5, np.zeros_like(image))
                assert len(outputs) == count, 'Blank image produced confident zero motion'
                send(6, image[:180, :320])
                assert len(outputs) == count, 'Uncalibrated resolution produced odometry'
                send(7, image)
                send(8, shifted)
                wait_until(node, lambda: len(outputs) > count)
                count = len(outputs)
                send(8, shifted)
                assert len(outputs) == count, 'Duplicate image timestamp produced odometry'
                send(7, image)
                assert len(outputs) == count, 'Backward image timestamp produced odometry'
                send(20, shifted)
                assert len(outputs) == count, 'Excessive image gap produced odometry'
                print('Node contract passed: metric velocity, frames, acquisition stamp, covariance, '
                      'unknown scale, stale orientation, blank frames, resolution mismatch, '
                      'recovery, duplicate/backward timestamps, excessive image gaps.')
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read(), file=sys.stderr)
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.close()
                wait_until(node, lambda: images.get_subscription_count() == 0)
    node.destroy_subscription(sub)
    node.destroy_subscription(preview_sub)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
