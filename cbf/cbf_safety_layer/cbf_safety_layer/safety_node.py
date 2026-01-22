"""Define safety layer to prevent collision."""
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
import pinocchio as pin
import os
from sensor_msgs.msg import JointState

import numpy as np

from cbf_safety_interfaces.msg import Constraint

import proxsuite

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
        self.links = ['fer_link4', 'fer_link7', 'fer_hand']
        self.frame_id = [self.model.getFrameId(n) for n in self.links]

        # Dimensions
        self.nq = self.model.nq  # size of joint angles vector
        self.nv = self.model.nv  # size of joint speed vector

        # CBF Parameters
        self.alpha = 10.0
        self.safety_margin = 0.05

        # # State variables
        # self.q = np.zeros(self.model.nq)  # Joint angles
        # self.v = np.zeros(self.model.nv)  # Joint velocity

        # Proxsuite solver setup
        # Equality constraint:0 , inequality: 1F
        n_constraints = len(self.links)
        self.qp = proxsuite.proxqp.dense.QP(self.nv, 0, n_constraints)
        self.qp.settings.eps_abs = 1.0e-5
        self.qp.settings.verbose = False
        self.qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT

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
        self.q = np.zeros(self.nq)
        self.get_logger().info('Safety Node and QP Initialized')

    def js_cb(self, msg):
        """
        Read input v_des and solve with QP to output v_safe.

        :param msg: msg from the joint state topic
        """
        # Input
        input_q = np.array(msg.position)
        if len(msg.velocity) > 0:
            input_v = np.array(msg.velocity)
        else:
            input_v = np.zeros(len(input_q))

        if len(input_q) == 7 and self.nq == 9:
            self.q = np.concatenate([input_q, [0, 0]])
            v_des = np.concatenate([input_v, [0, 0]])
        else:
            self.q = input_q
            v_des = input_v

        # calculate pos and ori of every joint relative to world
        # input : q (angles), output: joint positions
        pin.forwardKinematics(self.model, self.data, self.q, v_des)
        # calculate location of frame attached to joint
        # input: self.data, output: hand position(x,y,z)
        pin.updateFramePlacements(self.model, self.data)
        # calculate Jacobian of the robot
        pin.computeJointJacobians(self.model, self.data, self.q)

        # Cost Function
        H = np.eye(self.nv)
        g = -v_des

        # Lists for stacked constraints
        C_list = []
        l_list = []
        u_list = []

        for f_id in self.frame_id:
            # h = z_current - safety_margin
            h = self.data.oMf[f_id].translation[2] - self.safety_margin
            J = pin.getFrameJacobian(self.model, self.data, f_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            grad_h = J[2, :]  # Z-row only

            C_list.append(grad_h.reshape(1, self.nv))
            l_list.append(np.array([-self.alpha * h]))
            u_list.append(np.array([np.inf]))

        C = np.vstack(C_list)
        lower = np.concatenate(l_list)
        upper = np.concatenate(u_list)

        # --- D. SOLVE ---
        # Pass the stacked matrices to the solver
        self.qp.init(H, g, np.zeros((0, self.nv)), np.zeros(0),
                     C, lower, upper)
        self.qp.solve()

        # Extract the optimal velocity
        v_safe = self.qp.results.x

        dt = 0.1
        q_safe = self.q + v_safe*dt

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
