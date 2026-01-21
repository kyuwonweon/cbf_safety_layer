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
        urdf_dir = get_package_share_directory('franka_description')
        urdf_path = os.path.join(urdf_dir, 'robots/fer/fer.urdf')

        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.hand_id = self.model.getFrameId('fer_hand')
        self.elbow_id = self.model.getFrameId('fer_link4')

        # CBF Parameters
        self.gamma = 5.0
        self.safety_margin = 0.05
        # State variables
        self.q = np.zeros(self.model.nq)
        self.v = np.zeros(self.model.nv)

        self.js_sub = self.create_subscription(JointState,
                                               '/joint_states',
                                               self.js_cb,
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
            self.v = np.arra(msg.velocity)
        else:
            self.v = np.zeros(self.model.nv)
        pin.forwardKinematics(self.model, self.data, self.q, self.v)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, self.q)

        # Worksapce Constraint
        # Prevent hitting the table by setting safety margin(h)
        current_z = self.data.oMf[self.hand_id].translation[2]
        h = current_z - self.safety_margin

        J = pin.getFrameJacobian(self.model, self.data, self.hand_id,
                                 pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        grad_h = J[2, :]

        h_dot = grad_h@self.v
        max_h_dot = -self.gamma*h

        buffer = h_dot - max_h_dot

        if buffer < 0:
            self.get_logger().info('COLLIDED')

        msg = Constraint()
        msg.value = h
        msg.gradient = grad_h.tolist()
        self.constraint_pub.publish(msg)


def main(args=None):
    """Spin the node."""
    rclpy.init(args=args)
    node = SafetyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
