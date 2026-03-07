#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <franka_msgs/msg/franka_robot_state.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <interactive_markers/interactive_marker_server.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/bool.hpp>

#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <optional>
#include <vector>
#include <mutex>
#include <algorithm>

using namespace std::chrono_literals;

struct Capsule {
    std::string start_frame;
    std::string end_frame;
    double radius;
};

class SafetyNode : public rclcpp::Node {
public:
    SafetyNode() : Node("safety_node_cpp") {
        declare_parameter("self_robot_description",  rclcpp::ParameterValue(std::string("")));
        declare_parameter("other_robot_description", rclcpp::ParameterValue(std::string("")));
        declare_parameter("use_fallback_urdf",       rclcpp::ParameterValue(true));
        declare_parameter("fallback_urdf_package",   rclcpp::ParameterValue(std::string("franka_description")));
        declare_parameter("fallback_urdf_path",      rclcpp::ParameterValue(std::string("/robots/fer/fer.urdf")));
        declare_parameter("self_frame_prefix",       rclcpp::ParameterValue(std::string("")));
        declare_parameter("other_frame_prefix",      rclcpp::ParameterValue(std::string("")));
        declare_parameter("base_offset_x",       0.0);
        declare_parameter("base_offset_y",       0.0);
        declare_parameter("base_offset_z",       0.0);
        declare_parameter("other_base_offset_x", 0.0);
        declare_parameter("other_base_offset_y", 0.0);
        declare_parameter("other_base_offset_z", 0.0);
        declare_parameter("base_yaw",       0.0);
        declare_parameter("other_base_yaw", 0.0);

        base_offset       = Eigen::Vector3d::Zero();
        other_base_offset = Eigen::Vector3d::Zero();

        base_offset(0) = get_parameter("base_offset_x").as_double();
        base_offset(1) = get_parameter("base_offset_y").as_double();
        base_offset(2) = get_parameter("base_offset_z").as_double();
        other_base_offset(0) = get_parameter("other_base_offset_x").as_double();
        other_base_offset(1) = get_parameter("other_base_offset_y").as_double();
        other_base_offset(2) = get_parameter("other_base_offset_z").as_double();

        double y_self  = get_parameter("base_yaw").as_double();
        double y_other = get_parameter("other_base_yaw").as_double();
        R_self  = Eigen::AngleAxisd(y_self,  Eigen::Vector3d::UnitZ()).toRotationMatrix();
        R_other = Eigen::AngleAxisd(y_other, Eigen::Vector3d::UnitZ()).toRotationMatrix();

        RCLCPP_INFO(get_logger(), "Robot Base Offset: [%.2f, %.2f, %.2f] Yaw: %.4f",
            base_offset(0), base_offset(1), base_offset(2), y_self);

        std::string self_urdf, other_urdf;
        rclcpp::sleep_for(500ms);

        if (!get_parameter("self_robot_description", self_urdf) || self_urdf.empty()) {
            if (get_parameter("use_fallback_urdf").as_bool()) {
                RCLCPP_WARN(get_logger(), "self_robot_description empty. Using fallback URDF.");
                std::string pkg  = get_parameter("fallback_urdf_package").as_string();
                std::string path = get_parameter("fallback_urdf_path").as_string();
                pinocchio::urdf::buildModel(
                    ament_index_cpp::get_package_share_directory(pkg) + path, model_self);
            } else {
                RCLCPP_ERROR(get_logger(), "No self_robot_description and fallback disabled!");
                return;
            }
        } else {
            pinocchio::urdf::buildModelFromXML(self_urdf, model_self);
        }

        if (!get_parameter("other_robot_description", other_urdf) || other_urdf.empty()) {
            if (get_parameter("use_fallback_urdf").as_bool()) {
                RCLCPP_WARN(get_logger(), "other_robot_description empty. Using fallback URDF.");
                std::string pkg  = get_parameter("fallback_urdf_package").as_string();
                std::string path = get_parameter("fallback_urdf_path").as_string();
                pinocchio::urdf::buildModel(
                    ament_index_cpp::get_package_share_directory(pkg) + path, model_other);
            } else {
                RCLCPP_WARN(get_logger(), "No other_robot_description. Single robot mode.");
                model_other = model_self;
            }
        } else {
            pinocchio::urdf::buildModelFromXML(other_urdf, model_other);
        }

        data_self  = pinocchio::Data(model_self);
        data_other = pinocchio::Data(model_other);
        nq_self    = model_self.nq;
        nv_self    = model_self.nv;
        nq_other   = model_other.nq;
        nv_other   = model_other.nv;

        RCLCPP_INFO(get_logger(), "Self Robot Loaded:  nq=%d, nv=%d", nq_self,  nv_self);
        RCLCPP_INFO(get_logger(), "Other Robot Loaded: nq=%d, nv=%d", nq_other, nv_other);

        cbf_on = true;

        q_min    = Eigen::VectorXd(nv_self);
        q_max    = Eigen::VectorXd(nv_self);
        v_limit  = Eigen::VectorXd(nv_self);

        q_min   << -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0,  0.0;
        q_max   <<  2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973, 0.04, 0.04;
        v_limit <<  2.1750,  2.1750,  2.1750,  2.1750,  2.6100,  2.6100,  2.6100, 0.1,  0.1;

        // HARDWARE: tighter limits
        q_min   *= 0.95;
        q_max   *= 0.95;
        v_limit *= 0.10;

        q_safe          = Eigen::VectorXd::Zero(nq_self);
        q_other_robot   = Eigen::VectorXd::Zero(nq_other);
        v_safe          = Eigen::VectorXd::Zero(nv_self);
        v_safe_filtered = Eigen::VectorXd::Zero(nv_self);
        v_manual_command  = Eigen::VectorXd::Zero(nv_self);
        v_teleop_command  = Eigen::VectorXd::Zero(nv_self);
        v_other_robot     = Eigen::VectorXd::Zero(nv_other);

        std::string s_pre = get_parameter("self_frame_prefix").as_string();
        std::string o_pre = get_parameter("other_frame_prefix").as_string();
        if (s_pre.empty()) s_pre = "fer_";
        if (o_pre.empty()) o_pre = "fer_";

        joint_names.clear();
        for (int i = 1; i < model_self.njoints; ++i)
            joint_names.push_back(model_self.names[i]);

        RCLCPP_INFO(get_logger(), "Loaded %zu joints from model:", joint_names.size());
        for (const auto& name : joint_names)
            RCLCPP_INFO(get_logger(), "  - %s", name.c_str());

        capsules_self["base"]  = {s_pre + "link0", s_pre + "link1", 0.15};
        capsules_self["link1"] = {s_pre + "link1", s_pre + "link2", 0.12};
        capsules_self["link2"] = {s_pre + "link2", s_pre + "link3", 0.10};
        capsules_self["link3"] = {s_pre + "link3", s_pre + "link4", 0.10};
        capsules_self["link4"] = {s_pre + "link4", s_pre + "link5", 0.14};
        capsules_self["link5"] = {s_pre + "link5", s_pre + "link6", 0.10};
        capsules_self["link6"] = {s_pre + "link6", s_pre + "link7", 0.10};
        capsules_self["hand"]  = {s_pre + "link7", s_pre + "hand",  0.11};

        capsules_other["base"]  = {o_pre + "link0", o_pre + "link1", 0.15};
        capsules_other["link1"] = {o_pre + "link1", o_pre + "link2", 0.12};
        capsules_other["link2"] = {o_pre + "link2", o_pre + "link3", 0.10};
        capsules_other["link3"] = {o_pre + "link3", o_pre + "link4", 0.10};
        capsules_other["link4"] = {o_pre + "link4", o_pre + "link5", 0.14};
        capsules_other["link5"] = {o_pre + "link5", o_pre + "link6", 0.10};
        capsules_other["link6"] = {o_pre + "link6", o_pre + "link7", 0.10};
        capsules_other["hand"]  = {o_pre + "link7", o_pre + "hand",  0.11};

        #ifdef NDEBUG
            RCLCPP_INFO(this->get_logger(), "BUILD STATUS: RELEASE MODE (Optimized)");
        #else
            RCLCPP_WARN(this->get_logger(), "BUILD STATUS: DEBUG MODE (Slow!)");
        #endif

        auto qos = rclcpp::QoS(10);

        // HARDWARE: subscribe to FrankaRobotState
        sub_js_self = create_subscription<franka_msgs::msg::FrankaRobotState>(
            "/joint_states_source", qos,
            std::bind(&SafetyNode::joint_cb_self, this, std::placeholders::_1));

        sub_js_other = create_subscription<franka_msgs::msg::FrankaRobotState>(
            "/joint_states_source_other", qos,
            std::bind(&SafetyNode::joint_cb_other, this, std::placeholders::_1));

        pub_safe   = create_publisher<sensor_msgs::msg::JointState>("safety/joint_states", qos);
        pub_marker = create_publisher<visualization_msgs::msg::MarkerArray>("/safety_marker", qos);
        pub_cmd    = create_publisher<std_msgs::msg::Float64MultiArray>("/velocity_group_controller/commands", 10);

        sub_manual_command = create_subscription<sensor_msgs::msg::JointState>(
            "/manual_vel", 10,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(cmd_mutex);
                if (msg->velocity.size() == static_cast<size_t>(nv_self))
                    for (int i = 0; i < nv_self; ++i) v_manual_command(i) = msg->velocity[i];
            });

        sub_teleop_command = create_subscription<sensor_msgs::msg::JointState>(
            "/teleop_vel", 10,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(cmd_mutex);
                if (msg->velocity.size() == static_cast<size_t>(nv_self))
                    for (int i = 0; i < nv_self; ++i) v_teleop_command(i) = msg->velocity[i];
            });

        pub_obs_pose = create_publisher<geometry_msgs::msg::Point>("/shared_obstacle", 10);
        sub_obs_pose = create_subscription<geometry_msgs::msg::Point>("/shared_obstacle", 10,
            [this](const geometry_msgs::msg::Point::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(obs_mutex);
                obs_pose << msg->x, msg->y, msg->z;
            });

        sub_cbf_toggle = create_subscription<std_msgs::msg::Bool>("cbf_toggle", qos,
            [this](const std_msgs::msg::Bool::SharedPtr msg) {
                cbf_on = msg->data;
                RCLCPP_INFO(get_logger(), "CBF Safety Layer: %s", cbf_on ? "ON" : "OFF");
            });

        server = std::make_shared<interactive_markers::InteractiveMarkerServer>("obstacle_server", this);
        create_interactive_markers();
        marker_timer = this->create_wall_timer(std::chrono::seconds(1),
            [this]() { if (server) server->applyChanges(); });

        last_viz_time = this->get_clock()->now();
        RCLCPP_INFO(this->get_logger(), "Hardware Safety Node Ready");
    }

private:
    pinocchio::Model model_self, model_other;
    pinocchio::Data  data_self,  data_other;
    bool cbf_on;
    int nq_self, nv_self, nq_other, nv_other;
    int loop_count = 0;
    rclcpp::Time last_viz_time = rclcpp::Time(0, 0, RCL_ROS_TIME);

    Eigen::Vector3d base_offset, other_base_offset;
    Eigen::Matrix3d R_self, R_other;
    std::vector<std::string> joint_names;
    std::vector<Eigen::MatrixXd> C_rows;
    std::vector<double> l_vals;
    std::vector<double> u_vals;

    Eigen::VectorXd q_safe, v_safe, v_safe_filtered;
    Eigen::VectorXd v_manual_command, v_teleop_command, q_min, q_max, v_limit;
    Eigen::VectorXd q_other_robot, v_other_robot;
    Eigen::Vector3d obs_pose = {0.6, 0.0, 0.4};
    double obs_radius = 0.10, safety_margin = 0.02, alpha = 5.0, floor_height = 0.0;
    bool first_run = true, solver_initialized = false, other_robot_detected = false;
    rclcpp::Time last_time;

    std::map<std::string, Capsule> capsules_self, capsules_other;
    std::shared_ptr<proxsuite::proxqp::dense::QP<double>> qp;
    std::shared_ptr<interactive_markers::InteractiveMarkerServer> server;
    rclcpp::TimerBase::SharedPtr marker_timer;

    std::mutex q_other_mutex;
    std::mutex cmd_mutex;
    std::mutex obs_mutex;

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sub_cbf_toggle;
    // HARDWARE: FrankaRobotState subscriptions
    rclcpp::Subscription<franka_msgs::msg::FrankaRobotState>::SharedPtr sub_js_self, sub_js_other;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_manual_command, sub_teleop_command;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_safe;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_cmd;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr pub_obs_pose;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr sub_obs_pose;

    void create_interactive_markers() {
        auto add_axis = [](visualization_msgs::msg::InteractiveMarker& m,
                           double w, double x, double y, double z, std::string name) {
            visualization_msgs::msg::InteractiveMarkerControl c;
            c.orientation.w = w; c.orientation.x = x;
            c.orientation.y = y; c.orientation.z = z;
            c.name = name;
            c.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS;
            m.controls.push_back(c);
        };

        visualization_msgs::msg::InteractiveMarker obs_marker;
        obs_marker.header.frame_id = "base";
        obs_marker.name = "obstacle_marker";
        obs_marker.pose.position.x = obs_pose(0);
        obs_marker.pose.position.y = obs_pose(1);
        obs_marker.pose.position.z = obs_pose(2);
        obs_marker.scale = 0.3;

        visualization_msgs::msg::Marker sphere;
        sphere.type = visualization_msgs::msg::Marker::SPHERE;
        sphere.scale.x = sphere.scale.y = sphere.scale.z = obs_radius * 2.0;
        sphere.color.r = 1.0; sphere.color.a = 0.8;

        visualization_msgs::msg::InteractiveMarkerControl ctrl;
        ctrl.always_visible = true;
        ctrl.markers.push_back(sphere);
        obs_marker.controls.push_back(ctrl);
        add_axis(obs_marker, 1, 1, 0, 0, "move_x");
        add_axis(obs_marker, 1, 0, 1, 0, "move_z");
        add_axis(obs_marker, 1, 0, 0, 1, "move_y");
        server->insert(obs_marker);

        visualization_msgs::msg::InteractiveMarker f_marker;
        f_marker.header.frame_id = "base";
        f_marker.name = "floor_marker";
        f_marker.pose.position.z = floor_height;
        f_marker.scale = 0.5;

        visualization_msgs::msg::Marker plane;
        plane.type = visualization_msgs::msg::Marker::CUBE;
        plane.scale.x = 2.0; plane.scale.y = 2.0; plane.scale.z = 0.01;
        plane.color.b = 1.0; plane.color.a = 0.3;

        visualization_msgs::msg::InteractiveMarkerControl f_ctrl;
        f_ctrl.always_visible = true;
        f_ctrl.markers.push_back(plane);
        f_marker.controls.push_back(f_ctrl);
        add_axis(f_marker, 1, 0, 1, 0, "move_z");
        server->insert(f_marker);

        server->setCallback("obstacle_marker",
            std::bind(&SafetyNode::process_obs_feedback, this, std::placeholders::_1));
        server->setCallback("floor_marker",
            std::bind(&SafetyNode::process_floor_feedback, this, std::placeholders::_1));
        server->applyChanges();
    }

    void process_obs_feedback(
        const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr& feedback) {
        if (feedback->event_type !=
            visualization_msgs::msg::InteractiveMarkerFeedback::POSE_UPDATE) return;
        {
            std::lock_guard<std::mutex> lock(obs_mutex);
            obs_pose << feedback->pose.position.x,
                        feedback->pose.position.y,
                        feedback->pose.position.z;
        }
        geometry_msgs::msg::Point p;
        p.x = feedback->pose.position.x;
        p.y = feedback->pose.position.y;
        p.z = feedback->pose.position.z;
        pub_obs_pose->publish(p);
    }

    void process_floor_feedback(
        const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr& feedback) {
        if (feedback->event_type ==
            visualization_msgs::msg::InteractiveMarkerFeedback::POSE_UPDATE)
            floor_height = feedback->pose.position.z;
    }

    void joint_cb_other(const franka_msgs::msg::FrankaRobotState::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(q_other_mutex);
        size_t qlim = std::min(static_cast<size_t>(nq_other),
                               msg->measured_joint_state.position.size());
        for (size_t i = 0; i < qlim; ++i)
            q_other_robot[i] = msg->measured_joint_state.position[i];
        size_t vlim = std::min(static_cast<size_t>(nv_other),
                               msg->measured_joint_state.velocity.size());
        for (size_t i = 0; i < vlim; ++i)
            v_other_robot[i] = msg->measured_joint_state.velocity[i];
        if (!other_robot_detected) {
            other_robot_detected = true;
            RCLCPP_INFO(get_logger(), "Other robot detected - enabling inter-robot collision avoidance");
        }
    }

    void joint_cb_self(const franka_msgs::msg::FrankaRobotState::SharedPtr msg) {
        auto t_start = std::chrono::high_resolution_clock::now();
        auto now = this->get_clock()->now();

        if (first_run) {
            last_time     = now;
            last_viz_time = now;
            first_run     = false;
            C_rows.reserve(100);
            l_vals.reserve(100);
        }
        double dt = (now - last_time).seconds();
        last_time = now;

        static bool initial_pose_set = false;
        if (!initial_pose_set) {
            size_t lim = std::min(static_cast<size_t>(nq_self),
                                  msg->measured_joint_state.position.size());
            for (size_t i = 0; i < lim; ++i)
                q_safe[i] = msg->measured_joint_state.position[i];
            initial_pose_set = true;
        }

        pinocchio::forwardKinematics(model_self, data_self, q_safe);
        pinocchio::updateFramePlacements(model_self, data_self);
        pinocchio::computeJointJacobians(model_self, data_self, q_safe);
        pinocchio::crba(model_self, data_self, q_safe);
        data_self.M.triangularView<Eigen::StrictlyLower>() =
            data_self.M.transpose().triangularView<Eigen::StrictlyLower>();

        if (dt < 0.001) return;

        Eigen::VectorXd v_des;
        {
            std::lock_guard<std::mutex> lock(cmd_mutex);
            v_des = (v_teleop_command.norm() > 0.01) ? v_teleop_command : v_manual_command;
            v_des = v_des.cwiseMin(2.0).cwiseMax(-2.0);
        }

        Eigen::VectorXd q_other_copy, v_other_copy;
        bool other_robot_active = false;
        {
            std::lock_guard<std::mutex> lock(q_other_mutex);
            if (other_robot_detected) {
                q_other_copy     = q_other_robot;
                v_other_copy     = v_other_robot;
                other_robot_active = true;
            }
        }
        if (other_robot_active) {
            pinocchio::forwardKinematics(model_other, data_other, q_other_copy);
            pinocchio::updateFramePlacements(model_other, data_other);
            pinocchio::computeJointJacobians(model_other, data_other, q_other_copy);
        }

        Eigen::Vector3d obs_world;
        {
            std::lock_guard<std::mutex> lock(obs_mutex);
            obs_world = obs_pose;
        }
        // Obstacle from world/base frame into self robot-local frame
        Eigen::Vector3d obs_local = R_self.transpose() * (obs_world - base_offset);

        C_rows.clear();
        l_vals.clear();
        u_vals.clear();

        double ke     = (0.5 * v_safe.transpose() * data_self.M * v_safe).value();
        Eigen::RowVectorXd c_energy = v_safe.transpose() * data_self.M;
        double gamma  = 2.0;

        for (const auto& [name, cap] : capsules_self) {
            if (!model_self.existFrame(cap.start_frame) ||
                !model_self.existFrame(cap.end_frame)) {
                static std::set<std::string> warned_frames;
                if (warned_frames.find(name) == warned_frames.end()) {
                    RCLCPP_WARN(get_logger(), "Frame %s or %s not found",
                        cap.start_frame.c_str(), cap.end_frame.c_str());
                    warned_frames.insert(name);
                }
                continue;
            }

            auto id_s = model_self.getFrameId(cap.start_frame);
            auto id_e = model_self.getFrameId(cap.end_frame);
            Eigen::Vector3d p_s_raw = data_self.oMf[id_s].translation();
            Eigen::Vector3d p_e_raw = data_self.oMf[id_e].translation();

            Eigen::Vector3d p_s = p_s_raw, p_e = p_e_raw;
            Eigen::Vector3d dir = p_e_raw - p_s_raw;
            if (dir.norm() > 1e-4) {
                dir.normalize();
                p_s = p_s_raw - dir * 0.10;
                p_e = p_e_raw + dir * 0.10;
            }

            // Floor constraint
            if (p_e[2] < floor_height + 0.5) {
                double h_f = p_e[2] - cap.radius - floor_height - safety_margin;
                if (h_f < 0.15) {
                    Eigen::MatrixXd J_e(6, nv_self); J_e.setZero();
                    pinocchio::getFrameJacobian(model_self, data_self, id_e,
                        pinocchio::LOCAL_WORLD_ALIGNED, J_e);
                    Eigen::RowVectorXd J_floor = J_e.row(2);
                    double h_dot_f  = J_floor.dot(v_safe);
                    double h_f_safe = std::max(h_f, -0.05);
                    double b_floor  = (gamma * h_f_safe) - ke;
                    double u_floor  = (gamma * h_dot_f) + (alpha * b_floor);
                    if (h_f < 0) u_floor += 5.0 * std::abs(h_f);
                    if (v_safe.norm() > 0.01) {
                        C_rows.push_back(c_energy);
                        l_vals.push_back(-1e20);
                        u_vals.push_back(u_floor);
                    }
                    double kp = 20.0, kd = 5.0;
                    double l_kin = std::clamp(-kp * h_f - kd * h_dot_f, -10.0, 10.0);
                    C_rows.push_back(J_floor);
                    l_vals.push_back(l_kin);
                    u_vals.push_back(1e20);
                }
            }

            // Obstacle constraint
            auto pt  = closest_point_on_seg(p_s, p_e, obs_local);
            double d_p = (pt - obs_local).norm();
            if (d_p < 0.3) {
                double h_obs = d_p - cap.radius - obs_radius - safety_margin;
                if (h_obs < 0.2) {
                    Eigen::Vector3d n = (d_p > 1e-6) ?
                        ((pt - obs_local) / d_p) : Eigen::Vector3d(1, 0, 0);
                    Eigen::MatrixXd J_s(6, nv_self); J_s.setZero();
                    pinocchio::getFrameJacobian(model_self, data_self, id_s,
                        pinocchio::LOCAL_WORLD_ALIGNED, J_s);
                    Eigen::MatrixXd J_pt  = J_s.topRows(3) - skew(pt - p_s_raw) * J_s.bottomRows(3);
                    Eigen::RowVectorXd J_obs = n.transpose() * J_pt;
                    double h_dot_obs = J_obs.dot(v_safe);

                    double h_safe = std::max(h_obs, -0.05);
                    double b_obs  = (gamma * h_safe) - ke;
                    double u_obs  = (gamma * h_dot_obs) + (alpha * b_obs);
                    if (h_obs < 0) u_obs += 5.0 * std::abs(h_obs);
                    if (v_safe.norm() > 0.01) {
                        C_rows.push_back(c_energy);
                        l_vals.push_back(-1e20);
                        u_vals.push_back(u_obs);
                    }
                    double kp = 20.0, kd = 5.0;
                    double l_kin = std::clamp(-kp * h_obs - kd * h_dot_obs, -10.0, 10.0);
                    C_rows.push_back(J_obs);
                    l_vals.push_back(l_kin);
                    u_vals.push_back(1e20);
                }
            }

            // Robot-to-robot constraint
            if (other_robot_active) {
                for (const auto& [name_, cap_] : capsules_other) {
                    if (!model_other.existFrame(cap_.start_frame) ||
                        !model_other.existFrame(cap_.end_frame)) continue;
                    auto id_start = model_other.getFrameId(cap_.start_frame);
                    auto id_end   = model_other.getFrameId(cap_.end_frame);

                    Eigen::Vector3d p_s_other_raw = data_other.oMf[id_start].translation();
                    Eigen::Vector3d p_e_other_raw = data_other.oMf[id_end].translation();

                    // Transform other robot points from other-local -> world -> self-local
                    Eigen::Vector3d start_point = R_self.transpose() * (R_other * p_s_other_raw + other_base_offset - base_offset);
                    Eigen::Vector3d end_point   = R_self.transpose() * (R_other * p_e_other_raw + other_base_offset - base_offset);

                    Eigen::Vector3d dir_ = end_point - start_point;
                    if (dir_.norm() > 1e-4) {
                        dir_.normalize();
                        start_point -= dir_ * 0.1;
                        end_point   += dir_ * 0.1;
                    }

                    auto [pt_start, pt_end] = closest_segments_points(p_s, p_e, start_point, end_point);
                    double dist   = (pt_start - pt_end).norm();
                    double h_dyn  = dist - (cap.radius + cap_.radius + safety_margin);

                    if (dist < 0.6 && h_dyn < 0.3) {
                        Eigen::Vector3d n = (dist > 1e-5) ?
                            ((pt_start - pt_end) / dist) : Eigen::Vector3d(1, 0, 0);

                        Eigen::MatrixXd J_s(6, nv_self); J_s.setZero();
                        pinocchio::getFrameJacobian(model_self, data_self, id_s,
                            pinocchio::LOCAL_WORLD_ALIGNED, J_s);
                        Eigen::MatrixXd J_s_pt =
                            J_s.topRows(3) - skew(pt_start - p_s_raw) * J_s.bottomRows(3);
                        Eigen::RowVectorXd J_rel = n.transpose() * J_s_pt;

                        Eigen::MatrixXd J_o(6, nv_other); J_o.setZero();
                        pinocchio::getFrameJacobian(model_other, data_other, id_start,
                            pinocchio::LOCAL_WORLD_ALIGNED, J_o);

                        // pt_end is in self-local frame, transform back to other-local frame
                        Eigen::Vector3d pt_world      = R_self * pt_end + base_offset;
                        Eigen::Vector3d pt_other_local = R_other.transpose() * (pt_world - other_base_offset);
                        Eigen::MatrixXd J_o_pt = J_o.topRows(3) -
                            skew(pt_other_local - data_other.oMf[id_start].translation()) * J_o.bottomRows(3);

                        // Velocity of other robot's contact point, rotated into self frame
                        Eigen::Vector3d v_obs_vec_local = J_o_pt * v_other_copy;
                        Eigen::Vector3d v_obs_vec_self  = R_self.transpose() * R_other * v_obs_vec_local;
                        double v_obs_proj  = n.dot(v_obs_vec_self);
                        double h_dot_dyn   = J_rel.dot(v_safe) - v_obs_proj;

                        double h_dyn_safe  = std::max(h_dyn, -0.05);
                        double b_dyn       = (gamma * h_dyn_safe) - ke;
                        double u_dyn       = (gamma * h_dot_dyn) + (alpha * b_dyn);
                        if (h_dyn < 0) u_dyn += 5.0 * std::abs(h_dyn);
                        if (v_safe.norm() > 0.01) {
                            C_rows.push_back(c_energy);
                            l_vals.push_back(-1e20);
                            u_vals.push_back(u_dyn);
                        }
                        double kp = 20.0, kd = 5.0;
                        double l_kin = std::clamp(-kp * h_dyn - kd * h_dot_dyn, -10.0, 10.0);
                        C_rows.push_back(J_rel);
                        l_vals.push_back(l_kin);
                        u_vals.push_back(1e20);
                    }
                }
            }

            // Self-collision constraint
            for (const auto& [name_, cap_] : capsules_self) {
                if (name >= name_) continue;

                auto get_idx = [](const std::string& n) {
                    if (n == "base") return 0;
                    if (n == "hand") return 7;
                    if (n.size() > 4 && n.substr(0, 4) == "link") return n[4] - '0';
                    return 99;
                };
                if (std::abs(get_idx(name) - get_idx(name_)) <= 3) continue;

                auto id_s2 = model_self.getFrameId(cap_.start_frame);
                auto id_e2 = model_self.getFrameId(cap_.end_frame);
                Eigen::Vector3d p_s2_raw = data_self.oMf[id_s2].translation();
                Eigen::Vector3d p_e2_raw = data_self.oMf[id_e2].translation();

                auto [pt1, pt2] = closest_segments_points(p_s_raw, p_e_raw, p_s2_raw, p_e2_raw);
                double dist   = (pt1 - pt2).norm();
                double h_self = dist - (cap.radius + cap_.radius + safety_margin);

                if (dist < 0.5 && h_self < 0.03) {
                    if (h_self < 0.0) {
                        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 500,
                            "SELF-COLLISION: [%s] vs [%s] by %.4f m",
                            name.c_str(), name_.c_str(), std::abs(h_self));
                    }
                    Eigen::Vector3d n = (dist > 1e-5) ?
                        ((pt1 - pt2) / dist) : Eigen::Vector3d(1, 0, 0);

                    Eigen::MatrixXd J1(6, nv_self); J1.setZero();
                    pinocchio::getFrameJacobian(model_self, data_self, id_s,
                        pinocchio::LOCAL_WORLD_ALIGNED, J1);
                    Eigen::MatrixXd J1_pt = J1.topRows(3) - skew(pt1 - p_s_raw) * J1.bottomRows(3);

                    Eigen::MatrixXd J2(6, nv_self); J2.setZero();
                    pinocchio::getFrameJacobian(model_self, data_self, id_s2,
                        pinocchio::LOCAL_WORLD_ALIGNED, J2);
                    Eigen::MatrixXd J2_pt = J2.topRows(3) - skew(pt2 - p_s2_raw) * J2.bottomRows(3);

                    Eigen::RowVectorXd J_self_rel = n.transpose() * (J1_pt - J2_pt);
                    double h_dot_self = J_self_rel.dot(v_safe);
                    double kp = 20.0, kd = 5.0;
                    double l_kin = std::clamp(-kp * h_self - kd * h_dot_self, -2.0, 2.0);

                    C_rows.push_back(J_self_rel);
                    l_vals.push_back(l_kin);
                    u_vals.push_back(1e20);
                }
            }
        }

        const int max_c = 150;
        Eigen::MatrixXd C_t = Eigen::MatrixXd::Zero(max_c, nv_self);
        Eigen::VectorXd l_t = Eigen::VectorXd::Constant(max_c, -1e20);
        Eigen::VectorXd u_t = Eigen::VectorXd::Constant(max_c,  1e20);

        if (!cbf_on) { C_rows.clear(); l_vals.clear(); u_vals.clear(); }
        apply_limits(C_t, l_t, u_t, C_rows, l_vals, u_vals, dt);

        double kp_acc = 10.0;
        Eigen::VectorXd a_des = kp_acc * (v_des - v_safe);
        Eigen::MatrixXd H = Eigen::MatrixXd::Identity(nv_self, nv_self);

        auto t_qp_start = std::chrono::high_resolution_clock::now();
        try {
            if (!solver_initialized) {
                qp = std::make_shared<proxsuite::proxqp::dense::QP<double>>(nv_self, 0, max_c);
                qp->settings.eps_abs           = 1e-3;
                qp->settings.eps_rel           = 1e-3;
                qp->settings.max_iter          = 100;
                qp->settings.max_iter_in       = 10;
                qp->settings.check_duality_gap = false;
                qp->settings.verbose           = false;
                qp->init(H, -a_des, std::nullopt, std::nullopt, C_t, l_t, u_t);
                solver_initialized = true;
            } else {
                qp->update(std::nullopt, -a_des, std::nullopt, std::nullopt, C_t, l_t, u_t);
            }

            if (!C_t.allFinite()) {
                RCLCPP_ERROR(get_logger(), "NaN in constraints - emergency stop");
                v_safe.setZero();
            } else {
                qp->solve();
                auto qp_dur = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - t_qp_start).count();
                if (qp_dur > 10) {
                    RCLCPP_ERROR(get_logger(), "QP SLOW (%ld ms) - stopping", qp_dur);
                    v_safe.setZero();
                } else if (qp->results.info.status ==
                           proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
                    v_safe += qp->results.x * dt;
                } else {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 500,
                        "QP Failed status=%d", (int)qp->results.info.status);
                    v_safe *= 0.90;
                }
            }
            for (int i = 0; i < nv_self; ++i)
                v_safe(i) = std::clamp(v_safe(i), -v_limit(i), v_limit(i));

        } catch (...) {
            RCLCPP_ERROR(get_logger(), "QP exception - emergency stop");
            v_safe.setZero();
            solver_initialized = false;
        }

        // HARDWARE: slew rate limiter
        double max_step = 0.0015;
        for (int i = 0; i < nv_self; ++i) {
            double diff = v_safe(i) - v_safe_filtered(i);
            v_safe_filtered(i) += std::clamp(diff, -max_step, max_step);
        }

        std_msgs::msg::Float64MultiArray cmd;
        for (int i = 0; i < 7; ++i) cmd.data.push_back(v_safe_filtered[i]);
        pub_cmd->publish(cmd);

        q_safe += v_safe_filtered * dt;

        sensor_msgs::msg::JointState msg_out;
        msg_out.header.stamp    = now;
        msg_out.header.frame_id = "base";
        msg_out.name            = joint_names;
        msg_out.position.resize(nq_self);
        msg_out.velocity.resize(nv_self);
        for (int i = 0; i < nq_self; ++i) msg_out.position[i] = q_safe[i];
        for (int i = 0; i < nv_self; ++i) msg_out.velocity[i] = v_safe_filtered[i];
        pub_safe->publish(msg_out);

        loop_count++;
        if ((now - last_viz_time).seconds() > 0.033) {
            publish_markers();
            last_viz_time = now;
        }
    }

    void apply_limits(Eigen::MatrixXd& C, Eigen::VectorXd& L, Eigen::VectorXd& U,
                      const std::vector<Eigen::MatrixXd>& Cr,
                      const std::vector<double>& Lr,
                      const std::vector<double>& Ur, double dt) {
        C.setZero(); L.setConstant(-1e20); U.setConstant(1e20);
        int row = 0;
        for (size_t i = 0; i < Cr.size(); ++i) {
            if (row >= C.rows()) break;
            C.row(row) = Cr[i]; L(row) = Lr[i]; U(row) = Ur[i];
            row++;
        }
        double a_max = 2.0;
        for (int i = 0; i < nv_self; ++i) {
            if (row >= C.rows()) break;
            C(row, i) = 1.0;
            double a_v_max = ( v_limit(i) - v_safe(i)) / dt;
            double a_v_min = (-v_limit(i) - v_safe(i)) / dt;
            double a_q_max = (q_max(i) - q_safe(i) - v_safe(i) * dt) / (dt * dt);
            double a_q_min = (q_min(i) - q_safe(i) - v_safe(i) * dt) / (dt * dt);
            L(row) = std::max({-a_max, a_v_min, a_q_min});
            U(row) = std::min({ a_max, a_v_max, a_q_max});
            row++;
        }
    }

    Eigen::Vector3d closest_point_on_seg(
        Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d p) {
        Eigen::Vector3d ab = b - a;
        double len_sq = ab.squaredNorm();
        if (len_sq < 1e-6) return a;
        double t = std::clamp((p - a).dot(ab) / len_sq, 0.0, 1.0);
        return a + t * ab;
    }

    std::pair<Eigen::Vector3d, Eigen::Vector3d> closest_segments_points(
        Eigen::Vector3d p1, Eigen::Vector3d p2,
        Eigen::Vector3d p3, Eigen::Vector3d p4) {
        Eigen::Vector3d d1 = p2 - p1, d2 = p4 - p3, r = p1 - p3;
        double a = d1.squaredNorm(), e = d2.squaredNorm(), f = d2.dot(r);
        double s = 0.0, t = 0.0;
        if (a <= 1e-6 && e <= 1e-6) {
            s = t = 0.0;
        } else if (a <= 1e-6) {
            s = 0.0; t = std::clamp(f / e, 0.0, 1.0);
        } else {
            double c = d1.dot(r);
            if (e <= 1e-6) {
                t = 0.0; s = std::clamp(-c / a, 0.0, 1.0);
            } else {
                double b = d1.dot(d2), denom = a * e - b * b;
                s = (denom != 0.0) ? std::clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
                t = (b * s + f) / e;
                if (t < 0.0) { t = 0.0; s = std::clamp(-c / a, 0.0, 1.0); }
                else if (t > 1.0) { t = 1.0; s = std::clamp((b - c) / a, 0.0, 1.0); }
            }
        }
        return {p1 + s * d1, p3 + t * d2};
    }

    Eigen::Matrix3d skew(Eigen::Vector3d v) {
        Eigen::Matrix3d m;
        m << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
        return m;
    }

    void publish_markers() {
        visualization_msgs::msg::MarkerArray ma;
        int id = 1;
        for (const auto& [name, cap] : capsules_self) {
            if (!model_self.existFrame(cap.start_frame) ||
                !model_self.existFrame(cap.end_frame)) continue;
            
            Eigen::Vector3d p_start = data_self.oMf[model_self.getFrameId(cap.start_frame)].translation();
            Eigen::Vector3d p_end   = data_self.oMf[model_self.getFrameId(cap.end_frame)].translation();
            Eigen::Vector3d p_start_base = R_self * p_start + base_offset;
            Eigen::Vector3d p_end_base   = R_self * p_end   + base_offset;
            Eigen::Vector3d axis = p_end_base - p_start_base;
            double raw_len = axis.norm();

            visualization_msgs::msg::Marker m;
            m.header.frame_id = "base";
            m.header.stamp    = this->get_clock()->now();
            m.id     = id++;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.color.g = 1.0; m.color.a = 0.4;

            if (raw_len < 1e-4) {
                m.type = visualization_msgs::msg::Marker::SPHERE;
                m.pose.position.x = p_start_base.x();
                m.pose.position.y = p_start_base.y();
                m.pose.position.z = p_start_base.z();
                m.scale.x = m.scale.y = m.scale.z = cap.radius * 2.0;
                m.pose.orientation.w = 1.0;
            } else {
                Eigen::Vector3d center_base = (p_start_base + p_end_base) * 0.5;
                double vis_len = raw_len + 0.20;

                Eigen::Quaterniond q;
                q.setFromTwoVectors(Eigen::Vector3d::UnitZ(), axis);

                m.type = visualization_msgs::msg::Marker::CYLINDER;
                m.pose.position.x = center_base.x();
                m.pose.position.y = center_base.y();
                m.pose.position.z = center_base.z();
                m.pose.orientation.w = q.w(); m.pose.orientation.x = q.x();
                m.pose.orientation.y = q.y(); m.pose.orientation.z = q.z();
                m.scale.x = m.scale.y = cap.radius * 2.0;
                m.scale.z = vis_len;
            }
            ma.markers.push_back(m);
        }
        pub_marker->publish(ma);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SafetyNode>());
    rclcpp::shutdown();
    return 0;
}