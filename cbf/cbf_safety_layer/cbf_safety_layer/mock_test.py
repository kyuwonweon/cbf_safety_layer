"""Create Mock test to test the safety node."""
import time

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState


class MockTest(Node):
    """Test safety node logic."""

    def __init__(self):
        """Initialize mock test variables."""
        super().__init__('mock_test')
        # publish to jointstate
        self.pub_ = self.create_publisher(JointState, '/joint_states', 10)
        self.timer_ = self.create_timer(0.1, self.timer_callback)  # 10Hz
        self.get_logger().info('Mock Test Starting')

    def timer_callback(self):
        """Call function for timer."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f'fer_joint{i+1}' for i in range(7)] + \
                   ['fer_finger_joint1', 'fer_finger_joint2']
        t = time.time()
        # Ready Pose of franka
        q = [0.0, -0.7853981633974483, 0.0, -2.356194490192345,
             0.0, 1.5707963267948966, 0.7853981633974483, 0.0, 0.0]
        q[1] = 2*np.sin(t)
        msg.position = q

        v = [0.0] * 9
        v[1] = 2*np.cos(t)
        msg.velocity = v
        self.pub_.publish(msg)


def main(args=None):
    """Spin the node."""
    rclpy.init(args=args)
    node = MockTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
