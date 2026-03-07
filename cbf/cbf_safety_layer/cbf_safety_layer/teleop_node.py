import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from sensor_msgs.msg import JointState
import numpy as np

# Xbox controller mapping
AXIS_LEFT_LR = 0
AXIS_LEFT_UD = 1
AXIS_RIGHT_LR = 3
AXIS_RIGHT_UD = 4
AXIS_LT = 2
AXIS_RT = 5
AXIS_DPAD_UD = 7

BUTTON_A = 0

class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')
        self.sub_joy = self.create_subscription(Joy, '/joy', self.joy_cb, 10)
        
        # Publishing JointState to feed your C++ Safety Node
        self.pub_cmd = self.create_publisher(JointState, '/safety/input_joint_states', 10)

        self.linear_scale = 0.3
        self.angular_scale = 0.5
        
        # Sized back to 9 for the JointState names array
        self.target_v = np.zeros(9)
        self.current_v = np.zeros(9)
        self.max_accel_step = 0.015 

        self.timer = self.create_timer(0.02, self.timer_cb)
        self.get_logger().info("Xbox Teleop Ready! Hold 'A' to move.")

    def joy_cb(self, msg: Joy):
        new_target = np.zeros(9)
        DEADZONE = 0.35

        def clean(val):
            return 0.0 if abs(val) < DEADZONE else val

        if len(msg.buttons) > BUTTON_A and msg.buttons[BUTTON_A] == 1:
            if len(msg.axes) > AXIS_LEFT_LR:
                new_target[0] = clean(msg.axes[AXIS_LEFT_LR]) * self.linear_scale
            if len(msg.axes) > AXIS_LEFT_UD:
                new_target[1] = clean(msg.axes[AXIS_LEFT_UD]) * self.linear_scale
            if len(msg.axes) > AXIS_RIGHT_UD:
                new_target[3] = clean(msg.axes[AXIS_RIGHT_UD]) * self.angular_scale
            if len(msg.axes) > AXIS_RIGHT_LR:
                new_target[4] = clean(msg.axes[AXIS_RIGHT_LR]) * self.angular_scale
            
            if len(msg.axes) > AXIS_RT and len(msg.axes) > AXIS_LT:
                val_lt = 0.0 if abs((1.0 - msg.axes[AXIS_LT]) / 2.0) < 0.1 else (1.0 - msg.axes[AXIS_LT]) / 2.0
                val_rt = 0.0 if abs((1.0 - msg.axes[AXIS_RT]) / 2.0) < 0.1 else (1.0 - msg.axes[AXIS_RT]) / 2.0
                new_target[5] = (val_rt - val_lt) * self.angular_scale
                
            if len(msg.axes) > AXIS_DPAD_UD:
                new_target[7] = msg.axes[AXIS_DPAD_UD] * 0.05 
                new_target[8] = msg.axes[AXIS_DPAD_UD] * 0.05 

        self.target_v = new_target

    def timer_cb(self):
        diff = self.target_v - self.current_v
        step = np.clip(diff, -self.max_accel_step, self.max_accel_step)
        self.current_v += step

        cmd = JointState()
        cmd.name = [
            "fer_joint1", "fer_joint2", "fer_joint3", "fer_joint4",
            "fer_joint5", "fer_joint6", "fer_joint7",
            "fer_finger_joint1", "fer_finger_joint2"
        ]
        cmd.velocity = self.current_v.tolist()
        cmd.header.stamp = self.get_clock().now().to_msg()
        self.pub_cmd.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("TeleopNode stopped cleanly via Ctrl+C.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()