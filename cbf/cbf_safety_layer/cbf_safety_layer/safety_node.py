"""Define safety layer to prevent collision."""
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
import pinocchio as pin
import os
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray, InteractiveMarker, InteractiveMarkerControl
from interactive_markers.interactive_marker_server import InteractiveMarkerServer

from geometry_msgs.msg import Point

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
        n_constraints = 2*len(self.links)
        self.qp = proxsuite.proxqp.dense.QP(self.nv, 0, n_constraints)
        self.qp.settings.eps_abs = 1.0e-5
        self.qp.settings.verbose = False
        self.qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT

        # Obstacle
        self.obs_pose = np.array([0.5, 0.0, 0.4])

        self.obs_sub = self.create_subscription(Point, 'obstacle_pose',
                                                self.obs_cb, 10)

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
        self.marker_pub = self.create_publisher(MarkerArray,
                                                '/safety_marker',
                                                10)
        self.interactive_server = InteractiveMarkerServer(self,
                                                          'interactive_marker')

        int_marker = InteractiveMarker()
        int_marker.header.frame_id = 'base'
        int_marker.name = 'sphere_obs'
        int_marker.scale = 0.3
        int_marker.pose.position.x = self.obs_pose[0]
        int_marker.pose.position.y = self.obs_pose[1]
        int_marker.pose.position.z = self.obs_pose[2]

        visual = Marker()
        visual.type = Marker.SPHERE
        visual.scale.x = 0.2
        visual.scale.y = 0.2
        visual.scale.z = 0.2
        visual.color.r = 0.0
        visual.color.g = 1.0
        visual.color.b = 0.0
        visual.color.a = 1.0
        
        sphere_control = InteractiveMarkerControl()
        sphere_control.always_visible = True
        sphere_control.markers.append(visual)
        sphere_control.interaction_mode = InteractiveMarkerControl.MOVE_3D
        int_marker.controls.append(sphere_control)

        # X-Axis (Red)
        control_x = InteractiveMarkerControl()
        control_x.name = 'move_x'
        control_x.orientation.w = 1.0
        control_x.orientation.x = 1.0
        control_x.orientation.y = 0.0
        control_x.orientation.z = 0.0
        control_x.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control_x)

        # Y-Axis (Green)
        control_y = InteractiveMarkerControl()
        control_y.name = 'move_y'
        control_y.orientation.w = 1.0
        control_y.orientation.x = 0.0
        control_y.orientation.y = 1.0
        control_y.orientation.z = 0.0
        control_y.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control_y)

        # Z-Axis (Blue)
        control_z = InteractiveMarkerControl()
        control_z.name = 'move_z'
        control_z.orientation.w = 1.0
        control_z.orientation.x = 0.0
        control_z.orientation.y = 0.0
        control_z.orientation.z = 1.0
        control_z.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
        int_marker.controls.append(control_z)

        self.interactive_server.insert(int_marker,
                                       feedback_callback=self.fb_callback)
        self.interactive_server.applyChanges()

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
            # floor constraint
            # h = z_current - safety_margin
            h = self.data.oMf[f_id].translation[2] - self.safety_margin
            J = pin.getFrameJacobian(self.model, self.data, f_id,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            grad_h = J[2, :]  # Z-row only

            C_list.append(grad_h.reshape(1, self.nv))
            l_list.append(np.array([-self.alpha * h]))
            u_list.append(np.array([np.inf]))

        # Sphere obstacle
        obs_r = 0.15

        for f_id in self.frame_id:
            pos = self.data.oMf[f_id].translation
            diff = pos - self.obs_pose
            dist = np.linalg.norm(diff)
            h = dist - obs_r - self.safety_margin

            if dist > 1e-6:
                grad_h = diff/dist
            else:
                grad_h = np.array([1.0, 0.0, 0.0])

            J = pin.getFrameJacobian(self.model, self.data, f_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_trans = J[:3, :] 

            # Constraint
            con = grad_h @ J_trans
            C_list.append(con.reshape(1, self.nv))
            l_list.append(np.array([-self.alpha*h]))
            u_list.append(np.array([np.inf]))

        C = np.vstack(C_list)
        lower = np.concatenate(l_list)
        upper = np.concatenate(u_list)

        # solve
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

        # publish collision markers
        self.collision_markers()

    def collision_markers(self):
        """Publish markers for collision points and gradient."""
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        id_counter = 0

        for id_ in self.frame_id:
            pos = self.data.oMf[id_].translation
            # collision sphere
            sphere = Marker()
            sphere.header.frame_id = 'base'
            sphere.header.stamp = stamp
            sphere.id = id_counter
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD

            sphere.scale.x = 0.1
            sphere.scale.y = 0.1
            sphere.scale.z = 0.1

            sphere.color.r = 1.0
            sphere.color.g = 0.0
            sphere.color.b = 0.0
            sphere.color.a = 1.0

            sphere.pose.position.x = pos[0]
            sphere.pose.position.y = pos[1]
            sphere.pose.position.z = pos[2]

            marker_array.markers.append(sphere)
            id_counter += 1

            diff = pos - self.obs_pose
            dist = np.linalg.norm(diff)
            if dist > 1e-6:
                grad = diff/dist
            else:
                grad = np.array([1.0, 0.0, 0.0])

            arrow = Marker()
            arrow.header.frame_id = 'base'
            arrow.header.stamp = stamp
            arrow.id = id_counter
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD

            arrow.scale.x = 0.02
            arrow.scale.y = 0.05
            arrow.scale.z = 0.05

            arrow.color.r = 0.0
            arrow.color.g = 0.0
            arrow.color.b = 1.0
            arrow.color.a = 1.0

            start_pose = Point()
            start_pose.x, start_pose.y, start_pose.z = pos[0], pos[1], pos[2]

            end_pose = Point()
            end_pose.x = pos[0] + grad[0]*0.3
            end_pose.y = pos[1] + grad[1]*0.3
            end_pose.z = pos[2] + grad[2]*0.3

            arrow.points = [start_pose, end_pose]
            marker_array.markers.append(arrow)
            id_counter += 1

        self.marker_pub.publish(marker_array)

    def obs_cb(self, msg):
        """
        Update obstacel postion.

        :param msg: message
        """
        self.obs_pose = np.array([msg.x, msg.y, msg.z])

    def fb_callback(self, feedback):
        """
        Update target with feedback.

        :param feedback: feedback from control server
        """
        p = feedback.pose.position
        self.obs_pose = np.array([p.x, p.y, p.z])


def main(args=None):
    """Spin the node."""
    rclpy.init(args=args)
    node = SafetyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
