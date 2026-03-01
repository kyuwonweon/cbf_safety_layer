import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math
import time

class CyclicDriver(Node):
    def __init__(self):
        super().__init__('cyclic_driver')
        # Create publishers for both robots
        self.pub1 = self.create_publisher(JointState, '/robot1/safety/input_joint_states', 10)
        self.pub2 = self.create_publisher(JointState, '/robot2/safety/input_joint_states', 10)
        
        # Run the loop at 50 Hz
        self.timer = self.create_timer(0.02, self.timer_callback) 
        self.start_time = time.time()
        self.get_logger().info("Publishing cyclic sine waves to both robots... Press Ctrl+C to stop.")

    def timer_callback(self):
        t = time.time() - self.start_time
        
        # Calculate a smooth sine wave (Amplitude: 0.6 rad/s, Speed: 1.5)
        v = 0.6 * math.sin(t * 1.5)

        # Robot 1 swings inward (negative) while Robot 2 swings inward (positive)
        msg1 = JointState()
        msg1.velocity = [-v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        msg2 = JointState()
        msg2.velocity = [v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.pub1.publish(msg1)
        self.pub2.publish(msg2)

def main():
    rclpy.init()
    node = CyclicDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()