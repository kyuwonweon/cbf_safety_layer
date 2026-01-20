"""Define safety layer to prevent collision."""
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
import pinocchio as pin
import os
from sensor_msgs.msg import JointState

import numpy as np

from cbf_safety_interfaces.msg import Constraint


class safety_node(Node):
    """Create Safety layer by preventing safety layer."""

    def __init__(self):
        """Initialize the safety node class."""
        super().__init__('safetynode')
        urdf_dir = get_package_share_directory('fanka_description')
        urdf_path = os.path.join(urdf_dir, 'robots/panda_arm.urdf')

        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.hand_id = self.model.getFrameId('fer_hand')
        self.elbow_id = self.model.getFrameId('fer_link4')
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
        q = np.array(msg.position)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.computeFrameJacobian(self.model, self.data, q, self.hand_id,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

        # Worksapce Constraint
        # Prevent hitting the table by setting safety margin(h)
        height_margin = 0.05
        current_z = self.data.oMf[self.hand_id].translation[2]
        h = current_z - height_margin

        safety_msg = Constraint()
        safety_msg.value = h

        J_z = J[2, :]  # desired state
        safety_msg.gradient = J_z.tolist()
        self.constraint_pub.publish(safety_msg)
