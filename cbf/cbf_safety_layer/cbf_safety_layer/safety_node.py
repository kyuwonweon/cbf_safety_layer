"""Define safety layer to prevent collision."""
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
import pinocchio as pin
import os
from sensor_msgs.msg import JointState

import numpy as np

from cbf_safety_interfaces.msg import Constraint


class SafetyNode(Node):
    """Create Safety layer by preventing safety layer."""

    def __init__(self):
        """Initialize the safety node class."""
        super().__init__('safetynode')
        # Get franka robot urdf information
        urdf_dir = get_package_share_directory('franka_description')
        urdf_path = os.path.join(urdf_dir, 'robots/fer/fer.urdf')

        # Load physical robot info into Pinocchio
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.hand_id = self.model.getFrameId('fer_hand')
        self.elbow_id = self.model.getFrameId('fer_link4')

        # CBF Parameters
        self.alpha = 10.0
        self.safety_margin = 0.05

        # State variables
        self.q = np.zeros(self.model.nq)  # Joint angles
        self.v = np.zeros(self.model.nv)  # Joint velocity

        self.js_sub = self.create_subscription(JointState,
                                               '/joint_states',
                                               self.js_cb,
                                               10)
        self.safe_pub = self.create_publisher(JointState,
                                              '/safety/joint_states',
                                              10)
        self.constraint_pub = self.create_publisher(Constraint,
                                                    '/safety_constraint',
                                                    10)

    def js_cb(self, msg):
        """
        Call Joint State callback function.

        :param msg: msg from the joint state topic
        """
        self.q = np.array(msg.position)
        if len(msg.velocity) == self.model.nv:
            self.v = np.array(msg.velocity)
        else:
            self.v = np.zeros(self.model.nv)

        # calculate pos and ori of every joint relative to world
        # input : q (angles), output: joint positions
        pin.forwardKinematics(self.model, self.data, self.q, self.v)
        # calculate location of frame attached to joint
        # input: self.data, output: hand position(x,y,z)
        pin.updateFramePlacements(self.model, self.data)
        # calculate Jacobian of the robot
        pin.computeJointJacobians(self.model, self.data, self.q)

        # Worksapce Constraint
        # Prevent hitting the table by setting safety margin(h)
        current_z = self.data.oMf[self.hand_id].translation[2]
        h = current_z - self.safety_margin

        # Get jacobian of the hand
        J = pin.getFrameJacobian(self.model, self.data, self.hand_id,
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        # Z-row of jacobian
        grad_h = J[2, :]

        h_dot = grad_h @ self.v
        psi = h_dot + self.alpha*h

        if psi < 0:
            self.get_logger().info('COLLIDED')
            v_safe = - (grad_h*psi) / np.dot(grad_h, grad_h) + self.v
        else:
            v_safe = self.v

        dt = 0.1
        q_safe = self.q+v_safe*dt

        j_msg = JointState()
        j_msg.header = msg.header
        j_msg.name = msg.name
        j_msg.position = q_safe.tolist()
        j_msg.velocity = v_safe.tolist()

        self.safe_pub.publish(j_msg)

        c_msg = Constraint()
        c_msg.value = h
        c_msg.gradient = grad_h.tolist()
        self.constraint_pub.publish(c_msg)


def main(args=None):
    """Spin the node."""
    rclpy.init(args=args)
    node = SafetyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
