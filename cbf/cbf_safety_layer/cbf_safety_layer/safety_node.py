"""Define safety layer to prevent collision."""
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
import pinocchio as pin
from pinocchio import skew
import os
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray, InteractiveMarker, InteractiveMarkerControl
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

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

        self.nq = self.model.nq  # size of joint angles vector
        self.nv = self.model.nv  # size of joint speed vector

        # CBF Parameters
        self.alpha = 5.0
        self.safety_margin = 0.08
        self.loop_count = 0

        # Real safety parameter
        self.q_sim = None
        self.last_proc_time = self.get_clock().now()

        self.capsules = {
            'base': {'start': 'fer_link0',
                     'end': 'fer_link1',
                     'radius': 0.13},
            'upper_arm': {'start': 'fer_link1',
                          'end': 'fer_link3',
                          'radius': 0.12},
            'forearm': {'start': 'fer_link3',
                        'end': 'fer_link5',
                        'radius': 0.10},
            'wrist': {'start': 'fer_link5',
                      'end': 'fer_link7',
                      'radius': 0.09},
            'hand_base': {'start': 'fer_link7',
                          'end': 'fer_hand',
                          'radius': 0.10},
            'finger_left': {'start': 'fer_hand',
                            'end': 'fer_leftfinger',
                            'radius': 0.04},
            'finger_right': {'start': 'fer_hand',
                             'end': 'fer_rightfinger',
                             'radius': 0.04}
        }

        # Proxsuite solver setup
        self.n_max_constraints = 2*len(self.capsules)
        # Define QP solve
        # input: # of variables, # of equality constraints, # of inequality constraints
        self.qp = proxsuite.proxqp.dense.QP(self.nv, 0, self.n_max_constraints)
        self.qp.settings.eps_abs = 1.0e-5
        self.qp.settings.verbose = False
        # Define how solver begins search for optimal solution at teach time step
        # Warm start gives it calculation from last frame 
        self.qp.settings.initial_guess = proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
        # Initialize QP with dummy data 
        self.qp.init(
            np.eye(self.nv), np.zeros(self.nv), 
            None, None, 
            np.zeros((self.n_max_constraints, self.nv)), 
            np.full(self.n_max_constraints, -np.inf), 
            np.full(self.n_max_constraints, np.inf)
        )
        # Obstacle
        self.obs_pose = np.array([0.5, 0.0, 0.4])
        self.obs_r = 0.10

        self.cb_group = ReentrantCallbackGroup()

        self.obs_sub = self.create_subscription(Point, 'obstacle_pose',
                                                self.obs_cb, 10)

        self.js_sub = self.create_subscription(JointState,
                                               '/joint_states',
                                               self.js_cb,
                                               10,
                                               callback_group=self.cb_group)
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
        int_marker.pose.orientation.w = 1.0

        visual = Marker()
        visual.type = Marker.SPHERE
        visual.scale.x = self.obs_r * 2.0
        visual.scale.y = self.obs_r * 2.0
        visual.scale.z = self.obs_r * 2.0
        visual.color.r = 0.0
        visual.color.g = 1.0
        visual.color.b = 0.0
        visual.color.a = 0.7
        visual.pose.orientation.w = 1.0

        sphere_control = InteractiveMarkerControl()
        sphere_control.name = 'sphere_controller'
        sphere_control.always_visible = True
        sphere_control.markers.append(visual)
        sphere_control.interaction_mode = InteractiveMarkerControl.MOVE_3D
        sphere_control.orientation.w = 1.0

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

        int_marker.controls.append(sphere_control)

        self.interactive_server.insert(int_marker,
                                       feedback_callback=self.fb_callback)
        self.interactive_server.applyChanges()
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.q = np.zeros(self.nq)
        self.get_logger().info('Safety Node and QP Initialized')

    def timer_callback(self):
        """Implemeted to help with visual stuttering being 10 Hz."""
        self.visualize_capsules()
        self.collision_markers()

    def js_cb(self, msg):
        """Calculate safe velocity using CBF-QP."""
        # Dynamic dt calculation to handle lag in simulation
        now = self.get_clock().now()
        dt_nano = (now - self.last_proc_time).nanoseconds
        dt = dt_nano / 1e9
        if dt < 0.001:
            return
        dt = min(dt, 0.1)

        self.last_proc_time = now

        # Input
        input_q = np.array(msg.position)
        if len(msg.velocity) > 0:
            input_v = np.array(msg.velocity)
        else:
            np.zeros(len(input_q))

        if len(input_q) == 7 and self.nq == 9:
            q_target = np.concatenate([input_q, [0, 0]])
            v_input = np.concatenate([input_v, [0, 0]])
        else:
            q_target = input_q
            v_input = input_v
        
        if self.q_sim is None:
            self.q_sim = q_target.copy()

        # FeedForward Control to use input velocity
        # FeedBack Control to pull q_sim toward q_target
        Kp = 2.0
        v_des = v_input + Kp * (q_target - self.q_sim)
        v_des = np.clip(v_des, -2.5, 2.5)

        # Update q with safe state for Pinocchio
        self.q = self.q_sim

        # Kinematics
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, self.q)
        # Constraints
        C_full = np.zeros((self.n_max_constraints, self.nv))
        l_full = np.full(self.n_max_constraints, -np.inf)
        u_full = np.full(self.n_max_constraints, np.inf)
        obs_segment = (self.obs_pose, self.obs_pose)
        idx = 0  # Track which constraint row is being filled

        for name, capsule in self.capsules.items():
            # CHeck if frame is valid
            if not self.model.existFrame(capsule["end"]): 
                idx += 2  # Skip slots
                continue

            id_start = self.model.getFrameId(capsule['start'])
            id_end = self.model.getFrameId(capsule['end'])
            p_start = self.data.oMf[id_start].translation
            p_end = self.data.oMf[id_end].translation
            robot_radius = capsule['radius']
            # FLOOR CONSTRAINT
            h_floor = p_end[2] - robot_radius - self.safety_margin

            if h_floor < 0.1:  # Only activate when close
                # Get Jacobian for this specific frame
                J_frame = pin.getFrameJacobian(
                    self.model, self.data, id_end, 
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                C_full[idx] = J_frame[2, :]
                l_full[idx] = -self.alpha * h_floor
            idx += 1

            p_robot, p_obs = self.closest_point_between_segments((self.data.oMf[self.model.getFrameId(capsule["start"])].translation, p_end), obs_segment)
            diff_vec = p_robot - p_obs
            dist = np.linalg.norm(diff_vec)
            h_obs = dist - (robot_radius + self.obs_r + self.safety_margin)

            if h_obs < 0.4:  # Only activate when close
                # Normal vector pointing away from obstacle
                if dist > 1e-6:
                    n = diff_vec / dist
                else:
                    n = np.array([1.0, 0.0, 0.0])

                # Get Jacobian and project onto normal
                J_frame = pin.getFrameJacobian(
                    self.model, self.data, id_end,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )

                # Transport to collision point
                lever = p_robot - p_end
                J_point = J_frame[:3] - pin.skew(lever) @ J_frame[3:]
                J_projected = n.T @ J_point

                C_full[idx] = J_projected
                l_full[idx] = -self.alpha * h_obs
            idx += 1

        H = np.eye(self.nv)
        g = -v_des
        v_safe = v_des

        try:
            # Faster update than init for qp
            self.qp.update(H=H, g=g, C=C_full, l=l_full, u=u_full)
            self.qp.solve()
            if self.qp.results.info.status == proxsuite.proxqp.QPSolverOutput.PROXQP_SOLVED:
                v_safe = self.qp.results.x
            else:
                # Stop if fail
                v_safe = np.zeros(self.nv)
        except Exception:
            v_safe = np.zeros(self.nv)

        self.q_sim = self.q_sim + v_safe * dt

        j_msg = JointState()
        j_msg.header = msg.header
        j_msg.header.stamp = self.get_clock().now().to_msg()
        j_msg.name = msg.name

        if len(msg.name) == 7:
            j_msg.position = self.q_sim[:7].tolist()
            j_msg.velocity = v_safe[:7].tolist()
        else:
            j_msg.position = self.q_sim.tolist()
            j_msg.velocity = v_safe.tolist()

        self.safe_pub.publish(j_msg)

        # # Visualization (Every 10 frames)
        # self.loop_count += 1
        # if self.loop_count % 10 == 0:
        #     self.visualize_capsules()
        #     self.collision_markers()

    def collision_markers(self):
        """Publish markers for collision points and gradient."""
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        id_counter = 0

        # Iterate through capsules instead of undefined frame_id
        for name, capsule in self.capsules.items():
            if not self.model.existFrame(capsule["end"]):
                continue

            id_ = self.model.getFrameId(capsule["end"])
            pos = self.data.oMf[id_].translation
            # collision sphere
            sphere = Marker()
            sphere.ns = 'collisions'
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
            arrow.ns = 'gradients'
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

    def visualize_capsules(self):
        """Publish a MarkerArray for safety capsules."""
        marker_array = MarkerArray()
        timestamp = self.get_clock().now().to_msg()
        id_counter = 0

        for name, capsule in self.capsules.items():
            try:
                id_start = self.model.getFrameId(capsule['start'])
                id_end = self.model.getFrameId(capsule['end'])
                p_start = self.data.oMf[id_start].translation
                p_end = self.data.oMf[id_end].translation
            except KeyError:
                continue

            radius = capsule['radius']
            for p in [p_start, p_end]:
                sphere = Marker()
                sphere.header.frame_id = 'base'
                sphere.header.stamp = timestamp
                sphere.ns = 'capsules'
                sphere.id = id_counter
                id_counter += 1
                sphere.type = Marker.SPHERE
                sphere.action = Marker.ADD

                sphere.pose.position.x = p[0]
                sphere.pose.position.y = p[1]
                sphere.pose.position.z = p[2]

                sphere.scale.x = radius * 2.0
                sphere.scale.y = radius * 2.0
                sphere.scale.z = radius * 2.0

                sphere.color.r = 0.5
                sphere.color.g = 0.5
                sphere.color.b = 0.5
                sphere.color.a = 0.5
                marker_array.markers.append(sphere)

            vec = p_end - p_start
            length = np.linalg.norm(vec)

            if length > 1e-6:
                cyl = Marker()
                cyl.header.frame_id = 'base'
                cyl.header.stamp = timestamp
                cyl.ns = 'capsules'
                cyl.id = id_counter
                id_counter += 1
                cyl.type = Marker.CYLINDER
                cyl.action = Marker.ADD

                # Midpoint
                mid = (p_start + p_end) / 2.0
                cyl.pose.position.x = mid[0]
                cyl.pose.position.y = mid[1]
                cyl.pose.position.z = mid[2]

                # align Z-axis of cylinder with Vector
                q = self.get_orientation_between_vectors(np.array([0,0,1]), vec / length)
                cyl.pose.orientation.x = q[0]
                cyl.pose.orientation.y = q[1]
                cyl.pose.orientation.z = q[2]
                cyl.pose.orientation.w = q[3]
                cyl.scale.z = length
                cyl.scale.x = radius * 2.0
                cyl.scale.y = radius * 2.0
                cyl.color.r = 0.5
                cyl.color.g = 0.5
                cyl.color.b = 0.5
                cyl.color.a = 0.5
                marker_array.markers.append(cyl)
        self.marker_pub.publish(marker_array)

    def get_orientation_between_vectors(self, u, v):
        """Compute quaternion that rotates vector u to align with v."""
        # Fet axis of rotation
        axis = np.cross(u, v)
        norm_axis = np.linalg.norm(axis)
        # cosine of angle
        dot_product = np.dot(u, v)

        if norm_axis < 1e-6:
            if dot_product > 0:
                return [0.0, 0.0, 0.0, 1.0]  # Identity
            else:
                return [1.0, 0.0, 0.0, 0.0]  # 180 deg rotation
        # angle of rotation
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        # Create quaternion from Axis-Angle
        # q = [sin(theta/2)*axis, cos(theta/2)]
        sin_half = np.sin(angle / 2.0)
        axis_normalized = axis / norm_axis

        qx = axis_normalized[0] * sin_half
        qy = axis_normalized[1] * sin_half
        qz = axis_normalized[2] * sin_half
        qw = np.cos(angle / 2.0)
        return [qx, qy, qz, qw]

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

    def closest_point_between_segments(self, seg1, seg2):
        """Get closet point between segments."""
        p1, p2 = np.array(seg1[0]), np.array(seg1[1])
        p3, p4 = np.array(seg2[0]), np.array(seg2[1])
        v1 = p2 - p1
        v2 = p4 - p3
        gap = p1 - p3
        len_sq_v1 = np.dot(v1, v1)
        len_sq_v2 = np.dot(v2, v2)

        if len_sq_v2 < 1e-6:
            if len_sq_v1 > 1e-6:
                t_1 = np.clip(np.dot(-gap, v1) / len_sq_v1, 0.0, 1.0)
            else:
                t_1 = 0.0
            return p1 + t_1 * v1, p3

        seg_proj = np.dot(v1, v2)
        gap_proj_1 = np.dot(v1, gap)
        gap_proj_2 = np.dot(v2, gap)
        det = (len_sq_v1 * len_sq_v2) - (seg_proj**2)

        if det < 1e-6:  # Parallel
            t_1 = 0.0
            t_2 = np.clip(gap_proj_2 / len_sq_v2, 0.0, 1.0)
        else:  # Skew
            t_1 = (seg_proj * gap_proj_2 - gap_proj_1 * len_sq_v2) / det
            t_1 = np.clip(t_1, 0.0, 1.0)
            t_2 = (seg_proj * t_1 + gap_proj_1) / len_sq_v2
            if t_2 < 0.0:
                t_2 = 0.0
                t_1 = np.clip(-gap_proj_1 / len_sq_v1, 0.0, 1.0)
            elif t_2 > 1.0:
                t_2 = 1.0
                t_1 = np.clip((seg_proj - gap_proj_1) / len_sq_v1, 0.0, 1.0)

        return p1 + t_1 * v1, p3 + t_2 * v2


def main(args=None):
    """Spin the node."""
    rclpy.init(args=args)
    node = SafetyNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
