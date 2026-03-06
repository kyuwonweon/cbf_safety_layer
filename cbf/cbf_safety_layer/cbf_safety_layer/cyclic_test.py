import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import math
import time

class CyclicDriver(Node):
    def __init__(self):
        super().__init__('cyclic_driver')
        self.pub1 = self.create_publisher(JointState, '/robot1/manual_vel', 10)
        self.pub2 = self.create_publisher(JointState, '/robot2/manual_vel', 10)
        self.timer = self.create_timer(0.02, self.timer_callback) 
        self.start_time = time.time()
        self.get_logger().info("Publishing wide swing + elbow extension... Press Ctrl+C to stop.")

    def timer_callback(self):
        t = time.time() - self.start_time
        
        # Use a single, slightly slower sine wave so all joints peak together
        wave = math.sin(t * 0.5)
        
        v_base = 0.6 * wave       # Swing base inward

        msg1 = JointState()
        # [Base, Shoulder, Roll, Elbow, Wrist1, Wrist2, Wrist3, Finger1, Finger2]
        msg1.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        msg2 = JointState()
        msg2.velocity = [0.0, v_base, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

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