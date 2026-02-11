import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from sensor_msgs.msg import JointState
import numpy as np

# custom controller mapping to xbox controller
AXIS_LEFT_LR = 0   # Left Stick Left/Right -> Base Rotate
AXIS_LEFT_UD = 1   # Left Stick Up/Down    -> Shoulder
AXIS_RIGHT_LR = 3  # Right Stick Left/Right-> Wrist Flex
AXIS_RIGHT_UD = 4  # Right Stick Up/Down   -> Elbow
AXIS_LT = 2        # Left Trigger          -> Wrist Rotate (-)
AXIS_RT = 5        # Right Trigger         -> Wrist Rotate (+)
AXIS_DPAD_LR = 6   # D-Pad Left/Right      -> (Unused)
AXIS_DPAD_UD = 7   # D-Pad Up/Down         -> Gripper

BUTTON_A = 0       # 'A' Button should be pressed for commands to work


class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')
        self.sub_joy = self.create_subscription(Joy, '/joy', self.joy_cb, 10)
        self.pub_cmd = self.create_publisher(JointState,
                                             '/safety/input_joint_states', 10)

        self.linear_scale = 0.5   # Base/Shoulder Speed
        self.angular_scale = 1.0  # Wrist/Elbow Speed

        self.get_logger().info("Xbox Teleop Ready! Hold 'A' to move.")

    def joy_cb(self, msg: Joy):
        cmd = JointState()
        cmd.name = [
            "fer_joint1", "fer_joint2", "fer_joint3", "fer_joint4",
            "fer_joint5", "fer_joint6", "fer_joint7",
            "fer_finger_joint1", "fer_finger_joint2"
        ]

        v = np.zeros(9)
        DEADZONE = 0.25
        # Helper to apply deadzone

        def clean(val):
            return 0.0 if abs(val) < DEADZONE else val

        if len(msg.buttons) > BUTTON_A and msg.buttons[BUTTON_A] == 1:
            # Base (Joint 1)
            if len(msg.axes) > AXIS_LEFT_LR:
                v[0] = clean(msg.axes[AXIS_LEFT_LR]) * self.linear_scale

            # Shoulder (Joint 2)
            if len(msg.axes) > AXIS_LEFT_UD:
                v[1] = clean(msg.axes[AXIS_LEFT_UD]) * self.linear_scale

            # Elbow (Joint 4)
            if len(msg.axes) > AXIS_RIGHT_UD:
                v[3] = clean(msg.axes[AXIS_RIGHT_UD]) * self.angular_scale

            # Wrist Flex (Joint 5)
            if len(msg.axes) > AXIS_RIGHT_LR:
                v[4] = clean(msg.axes[AXIS_RIGHT_LR]) * self.angular_scale

            # Wrist Rotate (Joint 6)
            if len(msg.axes) > AXIS_RT:
                val_lt = (1.0 - msg.axes[AXIS_LT]) / 2.0 
                val_rt = (1.0 - msg.axes[AXIS_RT]) / 2.0
                if abs(val_lt) < 0.1:
                    val_lt = 0
                if abs(val_rt) < 0.1: 
                    val_rt = 0
                v[5] = (val_rt - val_lt) * self.angular_scale

            #  Grippers
            if len(msg.axes) > AXIS_DPAD_UD:
                v[7] = msg.axes[AXIS_DPAD_UD] * 0.05 
                v[8] = msg.axes[AXIS_DPAD_UD] * 0.05 

        cmd.velocity = v.tolist()
        cmd.header.stamp = self.get_clock().now().to_msg()
        self.pub_cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()