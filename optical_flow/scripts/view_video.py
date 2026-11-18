#!/usr/bin/env python3
"""Display BGR8 sensor_msgs/Image topics without copying pixels into Python bytes."""
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class Viewer(Node):
    def __init__(self):
        super().__init__('optical_flow_viewer')
        topic = self.declare_parameter('topic', '/bottom/flow/image').value
        self.latest = None
        self.latest_message = None
        self.dirty = False
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=2,
            reliability=ReliabilityPolicy.RELIABLE)
        self.subscription = self.create_subscription(Image, topic, self.receive, qos)
        self.timer = self.create_timer(1.0 / 60.0, self.display)

    def receive(self, msg):
        if (msg.encoding != 'bgr8' or msg.height <= 0 or msg.width <= 0
                or msg.step < msg.width * 3
                or len(msg.data) != msg.height * msg.step):
            self.get_logger().error('Expected a valid BGR8 sensor_msgs/Image')
            return
        # Keep the message alive while OpenCV uses its buffer; respect row padding.
        self.latest_message = msg
        self.latest = np.ndarray(
            shape=(msg.height, msg.width, 3), dtype=np.uint8,
            buffer=msg.data, strides=(msg.step, 3, 1))
        self.dirty = True

    def display(self):
        if self.dirty:
            cv2.imshow('Optical flow', self.latest)
            self.dirty = False
        if cv2.waitKey(1) & 0xff == ord('q'):
            rclpy.shutdown()


def main():
    rclpy.init()
    node = Viewer()
    cv2.namedWindow('Optical flow', cv2.WINDOW_NORMAL)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
