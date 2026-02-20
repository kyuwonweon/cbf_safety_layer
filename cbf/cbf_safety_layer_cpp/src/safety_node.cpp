#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <interactive_markers/interactive_marker_server.hpp> 
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <proxsuite/proxqp/dense/dense.hpp> 
#include <optional> 
#include <vector>
#include <mutex>

using namespace std::chrono_literals;

// Define Capsule shape for geometric primitve for the robot
struct Capsule {
    std::string start_frame;
    std::string end_frame;
    double radius;
};

class SafetyNode : public rclcpp::Node {
public:
    SafetyNode() : Node("safety_node_cpp") {
        // Declare parameters for both robots
        declare_parameter("self_robot_description", rclcpp::ParameterValue(std::string("")));
        declare_parameter("other_robot_description", rclcpp::ParameterValue(std::string("")));
        declare_parameter("use_fallback_urdf", rclcpp::ParameterValue(true));
        declare_parameter("fallback_urdf_package", rclcpp::ParameterValue(std::string("franka_description")));
        declare_parameter("fallback_urdf_path", rclcpp::ParameterValue(std::string("/robots/fer/fer.urdf")));
        declare_parameter("self_frame_prefix", rclcpp::ParameterValue(std::string("")));
        declare_parameter("other_frame_prefix", rclcpp::ParameterValue(std::string("")));
        declare_parameter("base_offset_x", 0.0);
        declare_parameter("base_offset_y", 0.0);
        declare_parameter("base_offset_z", 0.0);
        base_offset = Eigen::Vector3d::Zero();

        // get robot offsets from the "base" frame for rivz
        base_offset(0) = get_parameter("base_offset_x").as_double();
        base_offset(1) = get_parameter("base_offset_y").as_double();
        base_offset(2) = get_parameter("base_offset_z").as_double();

        std::string self_urdf, other_urdf;
        // Wait to ensure the urdf xml file is grabbed
        rclcpp::sleep_for(500ms); 
        
        // Load self robot model
        if (!get_parameter("self_robot_description", self_urdf) || self_urdf.empty()) {
            if (get_parameter("use_fallback_urdf").as_bool()) {
                RCLCPP_WARN(get_logger(), "Param 'self_robot_description' empty. Using fallback URDF.");
                std::string pkg = get_parameter("fallback_urdf_package").as_string();
                std::string path = get_parameter("fallback_urdf_path").as_string();
                std::string urdf_pkg = ament_index_cpp::get_package_share_directory(pkg);
                std::string urdf_path = urdf_pkg + path;
                pinocchio::urdf::buildModel(urdf_path, model_self);
            } else {
                RCLCPP_ERROR(get_logger(), "No self_robot_description provided and fallback disabled!");
                return;
            }
        } else {
            pinocchio::urdf::buildModelFromXML(self_urdf, model_self);
        }
        
        // Load other robot model
        if (!get_parameter("other_robot_description", other_urdf) || other_urdf.empty()) {
            if (get_parameter("use_fallback_urdf").as_bool()) {
                RCLCPP_WARN(get_logger(), "Param 'other_robot_description' empty. Using fallback URDF.");
                std::string pkg = get_parameter("fallback_urdf_package").as_string();
                std::string path = get_parameter("fallback_urdf_path").as_string();
                std::string urdf_pkg = ament_index_cpp::get_package_share_directory(pkg);
                std::string urdf_path = urdf_pkg + path;
                pinocchio::urdf::buildModel(urdf_path, model_other);
            } else {
                RCLCPP_WARN(get_logger(), "No other_robot_description provided. Single robot mode.");
                model_other = model_self;
            }
        } else {
            pinocchio::urdf::buildModelFromXML(other_urdf, model_other);
        }
        
        // -----------------------------------------------------------------------
        // Kinematic Data Intialization 
        // -----------------------------------------------------------------------
        data_self = pinocchio::Data(model_self);
        data_other = pinocchio::Data(model_other);
        nq_self = model_self.nq;
        nv_self = model_self.nv;
        nq_other = model_other.nq;
        nv_other = model_other.nv;
        
        RCLCPP_INFO(get_logger(), "Self Robot Loaded: nq=%d, nv=%d", nq_self, nv_self);
        RCLCPP_INFO(get_logger(), "Other Robot Loaded: nq=%d, nv=%d", nq_other, nv_other);

        // Initialize vectors
        q_min = Eigen::VectorXd(nv_self);
        q_max = Eigen::VectorXd(nv_self);
        v_limit = Eigen::VectorXd(nv_self);

        // Standard Franka Research 3 Limits
        q_min << -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0;
        q_max <<  2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973, 0.04, 0.04;
        v_limit << 2.1750,  2.1750,  2.1750,  2.1750,  2.6100,  2.6100,  2.6100, 0.1, 0.1;

        // Safety Buffer
        double padding = 0.02; 
        q_min = q_min.array() + padding;
        q_max = q_max.array() - padding;
        v_limit *= 0.95;

        // Setup empty vectors to hold safe and commanded velocity output
        q_safe = Eigen::VectorXd::Zero(nq_self);
        q_other_robot = Eigen::VectorXd::Zero(nq_other);
        v_safe = Eigen::VectorXd::Zero(nv_self);
        v_user_command = Eigen::VectorXd::Zero(nv_self);

        // -------------------------------------------------------
        // Setup Geometric primitve to encapsulate robot 
        // -------------------------------------------------------
        std::string s_pre = get_parameter("self_frame_prefix").as_string();
        std::string o_pre = get_parameter("other_frame_prefix").as_string();
        if (s_pre.empty()) s_pre = "fer_";
        if (o_pre.empty()) o_pre = "fer_";

        joint_names.clear();
        for (int i = 1; i < model_self.njoints; ++i) {
            joint_names.push_back(model_self.names[i]);
        }
        
        RCLCPP_INFO(get_logger(), "Loaded %zu joints from model:", joint_names.size());
        for (const auto& name : joint_names) {
            RCLCPP_INFO(get_logger(), " - %s", name.c_str());
        }
        
        // Self Capsules 
        capsules_self["base"]      = {s_pre + "link0", s_pre + "link1", 0.15};
        capsules_self["link1"]     = {s_pre + "link1", s_pre + "link2", 0.12};
        capsules_self["link2"]     = {s_pre + "link2", s_pre + "link3", 0.10};
        capsules_self["link3"]     = {s_pre + "link3", s_pre + "link4", 0.10};
        capsules_self["link4"]     = {s_pre + "link4", s_pre + "link5", 0.14};
        capsules_self["link5"]     = {s_pre + "link5", s_pre + "link6", 0.10};
        capsules_self["link6"]     = {s_pre + "link6", s_pre + "link7", 0.10};
        capsules_self["hand"]      = {s_pre + "link7", s_pre + "hand",  0.11};

        // Other Robot Capsules
        capsules_other["base"]      = {o_pre + "link0", o_pre + "link1", 0.15};
        capsules_other["link1"]     = {o_pre + "link1", o_pre + "link2", 0.12};
        capsules_other["link2"]     = {o_pre + "link2", o_pre + "link3", 0.10};
        capsules_other["link3"]     = {o_pre + "link3", o_pre + "link4", 0.10};
        capsules_other["link4"]     = {o_pre + "link4", o_pre + "link5", 0.14};
        capsules_other["link5"]     = {o_pre + "link5", o_pre + "link6", 0.10};
        capsules_other["link6"]     = {o_pre + "link6", o_pre + "link7", 0.10};
        capsules_other["hand"]      = {o_pre + "link7", o_pre + "hand",  0.11};

        // -------------------------------------------------------------
        // ROS 2 Communication diagnostics 
        //--------------------------------------------------------------
        // Check if the compilation is in release mode for faster computation 
        #ifdef NDEBUG
            RCLCPP_INFO(this->get_logger(), "BUILD STATUS: RELEASE MODE (Optimized)");
        #else
            RCLCPP_WARN(this->get_logger(), "BUILD STATUS: DEBUG MODE (Slow!)");
        #endif

        // --------------------------------------------------------------
        // ROS 2 Setup
        // --------------------------------------------------------------
        auto qos = rclcpp::QoS(10);
        
        // Joint state subscription for robots
        sub_js_self = create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_source", qos, std::bind(&SafetyNode::joint_cb_self, this, std::placeholders::_1));
        
        sub_js_other = create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_source_other", qos, std::bind(&SafetyNode::joint_cb_other, this, std::placeholders::_1));
        
        // 
        pub_safe = create_publisher<sensor_msgs::msg::JointState>("/safety/joint_states", qos);
        pub_marker = create_publisher<visualization_msgs::msg::MarkerArray>("/safety_marker", qos);
        pub_cmd = create_publisher<std_msgs::msg::Float64MultiArray>("/velocity_group_controller/commands", 10);

        sub_input_vel = create_subscription<sensor_msgs::msg::JointState>(
            "/safety/input_joint_states", 10, 
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(cmd_mutex);
                if (msg->velocity.size() == static_cast<size_t>(nv_self)) {
                    for(int i=0; i<nv_self; ++i) v_user_command(i) = msg->velocity[i];
                }
            });

        
        pub_obs_pose = create_publisher<geometry_msgs::msg::Point>("/shared_obstacle", 10);
        sub_obs_pose = create_subscription<geometry_msgs::msg::Point>("/shared_obstacle", 10,
        [this](const geometry_msgs::msg::Point::SharedPtr msg){
            std::lock_guard<std::mutex> lock(obs_mutex);
            obs_pose << msg->x, msg->y, msg->z;
        });
        
        // ------------------------------------------------------------------
        // Marker for Obstacle showing 
        // ------------------------------------------------------------------
        server = std::make_shared<interactive_markers::InteractiveMarkerServer>("obstacle_server", this);
        create_interactive_markers();
        marker_timer = this->create_wall_timer(
            std::chrono::seconds(1),
            [this]() { 
                if (server) server->applyChanges(); 
            });
        
        last_viz_time = this->get_clock()->now();
        RCLCPP_INFO(this->get_logger(), "Safety Node Ready");
    }

private:
    pinocchio::Model model_self, model_other;
    pinocchio::Data data_self, data_other;
    int nq_self, nv_self, nq_other, nv_other;
    int loop_count = 0;
    rclcpp::Time last_viz_time = rclcpp::Time(0, 0, RCL_ROS_TIME);
    rclcpp::Time last_obs_update = rclcpp::Time(0, 0, RCL_ROS_TIME);

    Eigen::Vector3d base_offset;
    std::vector<std::string> joint_names;
    std::vector<Eigen::MatrixXd> C_rows;
    std::vector<double> l_vals;
    
    Eigen::VectorXd q_safe, v_safe, v_user_command, q_min, q_max, v_limit, q_other_robot;
    Eigen::Vector3d obs_pose = {0.8, 0.0, 0.4}; 
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

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_js_self, sub_js_other, sub_input_vel;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_safe;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_cmd;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr pub_obs_pose;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr sub_obs_pose;

    void create_interactive_markers() {
        visualization_msgs::msg::InteractiveMarker obs_marker;
        obs_marker.header.frame_id = "base";
        obs_marker.name = "obstacle_marker";
        obs_marker.pose.position.x = obs_pose(0);
        obs_marker.pose.position.y = obs_pose(1);
        obs_marker.pose.position.z = obs_pose(2);
        obs_marker.scale = 0.3;

        visualization_msgs::msg::Marker sphere;
        sphere.type = visualization_msgs::msg::Marker::SPHERE;
        sphere.scale.x = obs_radius * 2.0;
        sphere.scale.y = obs_radius * 2.0;
        sphere.scale.z = obs_radius * 2.0;
        sphere.color.r = 1.0; sphere.color.a = 0.8;
        
        visualization_msgs::msg::InteractiveMarkerControl ctrl;
        ctrl.always_visible = true;
        ctrl.markers.push_back(sphere);
        obs_marker.controls.push_back(ctrl);

        auto add_axis = [&](visualization_msgs::msg::InteractiveMarker &m, double w, double x, double y, double z, std::string name) {
            visualization_msgs::msg::InteractiveMarkerControl c;
            c.orientation.w = w; c.orientation.x = x; c.orientation.y = y; c.orientation.z = z;
            c.name = name;
            c.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS;
            m.controls.push_back(c);
        };

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
        server->setCallback("obstacle_marker", std::bind(&SafetyNode::process_obs_feedback, this, std::placeholders::_1));
        server->setCallback("floor_marker", std::bind(&SafetyNode::process_floor_feedback, this, std::placeholders::_1));
        server->applyChanges();
    }

    void process_obs_feedback(const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr &feedback) {
        if (feedback->event_type != 
        visualization_msgs::msg::InteractiveMarkerFeedback::POSE_UPDATE) return;

    // Update obs_pose under lock, then release before publishing
    {
        std::lock_guard<std::mutex> lock(obs_mutex);
        obs_pose << feedback->pose.position.x, 
                    feedback->pose.position.y, 
                    feedback->pose.position.z;
        } // lock released here

        // Publish outside the lock — no deadlock risk
        geometry_msgs::msg::Point p;
        p.x = feedback->pose.position.x;
        p.y = feedback->pose.position.y;
        p.z = feedback->pose.position.z;
        pub_obs_pose->publish(p);
    }
    

    void process_floor_feedback(const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr &feedback) {
        if (feedback->event_type == visualization_msgs::msg::InteractiveMarkerFeedback::POSE_UPDATE) {
            floor_height = feedback->pose.position.z;
        }
    }

    void joint_cb_other(const sensor_msgs::msg::JointState::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(q_other_mutex);
        for (size_t i = 0; i < std::min(static_cast<size_t>(nq_other), msg->position.size()); ++i) {
            q_other_robot[i] = msg->position[i];
        }
        if (!other_robot_detected) {
            other_robot_detected = true;
            RCLCPP_INFO(this->get_logger(), "Other robot detected - enabling inter-robot collision avoidance");
        }
    }

    void joint_cb_self(const sensor_msgs::msg::JointState::SharedPtr msg) {
        // Profiling how long code takes at each section (local execution timing)
        auto t_start = std::chrono::high_resolution_clock::now();
        // ROS 2 Time Acquisition
        auto now = this->get_clock()->now();
        
        if (first_run) {
            last_time = now; // Ensure dt calculation is accurate 
            last_viz_time = now; 
            first_run = false;
            // Memory optimization to prevent constant resizing of vectors 
            C_rows.reserve(100);
            l_vals.reserve(100);
        }
        double dt = (now - last_time).seconds();
        last_time = now;

        // initialize robot position by getting real robot position than all zero default state
        static bool initial_pose_set = false;
        if (!initial_pose_set) {
             for(size_t i = 0; i < std::min(static_cast<size_t>(nq_self), msg->position.size()); ++i) {
                q_safe[i] = msg->position[i];
            }
            initial_pose_set = true;
        }

        // Update Kinematics 
        pinocchio::forwardKinematics(model_self, data_self, q_safe);
        pinocchio::updateFramePlacements(model_self, data_self);
        

        publish_markers();
        
        if (dt < 0.001) return;

        Eigen::VectorXd v_des;
        {
            std::lock_guard<std::mutex> lock(cmd_mutex);
            v_des = v_user_command.cwiseMin(2.0).cwiseMax(-2.0);
        }

        auto t_kin_start = std::chrono::high_resolution_clock::now();
        // Create Jacobian Matrix for the current safet joint position to become Cartesian velocities
        pinocchio::computeJointJacobians(model_self, data_self, q_safe);
        auto t_kin_end = std::chrono::high_resolution_clock::now();

        // Copy position of other robot
        Eigen::VectorXd q_other_copy;
        bool other_robot_active = false;
        {
            std::lock_guard<std::mutex> lock(q_other_mutex);
            if (other_robot_detected) {
                q_other_copy = q_other_robot;
                other_robot_active = true;
            }
        }

        // Shift obstacle position from the "world" coordinate relative to robot base
        Eigen::Vector3d obs_world;
        {
            std::lock_guard<std::mutex> lock(obs_mutex);
            obs_world = obs_pose;
        }
        Eigen::Vector3d obs_local = obs_world - base_offset;

        C_rows.clear();
        l_vals.clear();
        
        // Optimize QP solver time by running collision math only when things are within 1.5m
        bool constraints_needed = false;
        if (!capsules_self.empty()) {
            if (obs_local.norm()<1.5 || other_robot_active){
                constraints_needed = true;
            }
        }

        // If obstacle is far enough, then skip collision solver and do basic joint limit check
        if (!constraints_needed) {
            v_safe = v_des;
            for (int i=0; i<nv_self; ++i) {
                double v_min = std::max(-v_limit(i), (q_min(i) - q_safe(i)) / dt);
                double v_max = std::min(v_limit(i), (q_max(i) - q_safe(i)) / dt);
                v_safe(i) = std::clamp(v_safe(i), v_min, v_max);
            }
            
            // send clamped final safe velocity to robot
            std_msgs::msg::Float64MultiArray cmd;
            for(int i=0; i<7; ++i) cmd.data.push_back(v_safe[i]);
            pub_cmd->publish(cmd);
            
            // Update safe position of the robot to ROS and RVIZ
            q_safe += v_safe * dt;
            sensor_msgs::msg::JointState msg_out;
            msg_out.header.stamp = now;
            msg_out.header.frame_id = "base";
            msg_out.name = joint_names;
            msg_out.position.resize(nq_self); 
            msg_out.velocity.resize(nv_self);
            for(int i=0; i<nq_self; ++i) msg_out.position[i] = q_safe[i];
            for(int i=0; i<nv_self; ++i) msg_out.velocity[i] = v_safe[i];
            pub_safe->publish(msg_out);
            
            return; 
        }

        //debug
        double min_dist_found = 100.0;
        std::string closest_link = "";

        for (const auto& [name, cap] : capsules_self) {
            if (!model_self.existFrame(cap.start_frame) || !model_self.existFrame(cap.end_frame)) {
                // Warning only once to avoid spam
                static std::set<std::string> warned_frames;
                if (warned_frames.find(name) == warned_frames.end()) {
                     RCLCPP_WARN(this->get_logger(), "Frame %s or %s not found", cap.start_frame.c_str(), cap.end_frame.c_str());
                     warned_frames.insert(name);
                }
                continue;
            }

            auto id_s = model_self.getFrameId(cap.start_frame);
            auto id_e = model_self.getFrameId(cap.end_frame);
            Eigen::Vector3d p_s_raw = data_self.oMf[id_s].translation();
            Eigen::Vector3d p_e_raw = data_self.oMf[id_e].translation();
            
            Eigen::Vector3d p_s = p_s_raw;
            Eigen::Vector3d p_e = p_e_raw;
            Eigen::Vector3d dir = p_e_raw - p_s_raw;
            
            if (dir.norm() > 1e-4) {
                dir.normalize();
                double extension = 0.10; // The 10cm extension on both sides
                p_s = p_s_raw - (dir * extension);
                p_e = p_e_raw + (dir * extension);
            }
            
            // Floor Constraint - only check if close
            if (p_e[2] < floor_height + 0.5) {
                double h_f = p_e[2] - cap.radius - floor_height - safety_margin;
                if (h_f < 0.15) {
                    Eigen::MatrixXd J_e(6, nv_self);
                    pinocchio::getFrameJacobian(model_self, data_self, id_e, pinocchio::LOCAL_WORLD_ALIGNED, J_e);
                    C_rows.push_back(J_e.row(2));
                    l_vals.push_back(-alpha * std::max(h_f, 0.001));
                }
            }

            // Obstacle Constraint - only if close
            auto pt = closest_point_on_seg(p_s, p_e, obs_local);
            double d_p = (pt - obs_local).norm();
            double surface_dist = d_p - cap.radius - obs_radius;

            if (surface_dist<min_dist_found){
                min_dist_found = surface_dist;
                closest_link = name;
            }

            if (d_p < 0.5) {
                double h_p = d_p - (cap.radius + obs_radius + safety_margin);
                if (h_p < 0.25) {
                    add_constraint(C_rows, l_vals, pt, obs_local, d_p, h_p, id_s, p_s);
                }
            }
        }

        // if (loop_count % 100 == 0) { 
        //     RCLCPP_INFO(get_logger(), 
        //         "[%s] OBS DEBUG: World=[%.2f, %.2f] | Offset=[%.2f, %.2f] | Local=[%.2f, %.2f] | Closest Link='%s' Dist=%.3fm",
        //         get_namespace(),
        //         obs_world(0), obs_world(1),
        //         base_offset(0), base_offset(1),
        //         obs_local(0), obs_local(1),
        //         closest_link.c_str(), min_dist_found);
        // }

        auto t_constr_end = std::chrono::high_resolution_clock::now();

        const int max_c = 60;
        Eigen::MatrixXd C_t = Eigen::MatrixXd::Zero(max_c, nv_self); //constraint
        Eigen::VectorXd l_t = Eigen::VectorXd::Constant(max_c, -1e20); //lower limit
        Eigen::VectorXd u_t = Eigen::VectorXd::Constant(max_c, 1e20); //upper limit 

        apply_limits(C_t, l_t, u_t, C_rows, l_vals, dt);

        auto t_qp_start = std::chrono::high_resolution_clock::now();

        try {
            if (!solver_initialized) {
                // QP Solver initialization 
                qp = std::make_shared<proxsuite::proxqp::dense::QP<double>>(nv_self, 0, max_c);
                qp->settings.eps_abs = 1e-3;
                qp->settings.eps_rel = 1e-3;
                qp->settings.max_iter = 100;
                qp->settings.max_iter_in = 10;
                qp->settings.check_duality_gap = false;
                qp->settings.verbose = false;
                qp->settings.initial_guess = proxsuite::proxqp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
                
                // Goal: minimize difference between v_safe and v_des 
                qp->init(Eigen::MatrixXd::Identity(nv_self, nv_self), -v_des, std::nullopt, std::nullopt, C_t, l_t, u_t);
                solver_initialized = true;
            } else {
                // argument in order of (H, g, A, b, C, l, u)
                // H: Hessian >> no value as goal didn't change
                // Ax = b (Equality constraint) > no equality constraint so zero 
                qp->update(std::nullopt, -v_des, std::nullopt, std::nullopt, C_t, l_t, u_t);
            }
            
            // Shafety Check and Execution 
            if (!C_t.allFinite()) { 
                RCLCPP_ERROR(this->get_logger(), "NaN detected in Constraints!");
                v_safe.setZero();
            } else {
                qp->solve();
                
                // Timeout protection. If the solver doesn't answer within 10ms, stop the robot
                auto qp_dur = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - t_qp_start).count();
                if (qp_dur > 10) {
                    RCLCPP_ERROR(this->get_logger(), "QP SLOW (%ld ms)!", qp_dur);
                    v_safe.setZero();
                } else {
                    v_safe = (qp->results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) ? 
                             qp->results.x : Eigen::VectorXd::Zero(nv_self);
                }
            }
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(), "QP solver exception - emergency stop");
            v_safe.setZero();
            solver_initialized = false;
        }

        auto t_qp_end = std::chrono::high_resolution_clock::now();

        auto d_kin = std::chrono::duration_cast<std::chrono::microseconds>(t_kin_end - t_kin_start).count();
        auto d_con = std::chrono::duration_cast<std::chrono::microseconds>(t_constr_end - t_kin_end).count();
        auto d_qp  = std::chrono::duration_cast<std::chrono::microseconds>(t_qp_end - t_qp_start).count();
        auto d_tot = std::chrono::duration_cast<std::chrono::microseconds>(t_qp_end - t_start).count();

        if (d_tot > 1000) {
            RCLCPP_WARN(this->get_logger(), "TIMING: Total=%ld us | Kin=%ld | Cons=%ld | QP=%ld", d_tot, d_kin, d_con, d_qp);
        }

        // Pulls joint values(not the fingers) and sent them to velocity controller 
        std_msgs::msg::Float64MultiArray cmd;
        for(int i=0; i<7; ++i) cmd.data.push_back(v_safe[i]);
        pub_cmd->publish(cmd);

        q_safe += v_safe * dt;
        
        sensor_msgs::msg::JointState msg_out;
        msg_out.header.stamp = now;
        msg_out.header.frame_id = "base";
        msg_out.name = joint_names;
        msg_out.position.resize(nq_self); 
        msg_out.velocity.resize(nv_self);
        for(int i=0; i<nq_self; ++i) msg_out.position[i] = q_safe[i];
        for(int i=0; i<nv_self; ++i) msg_out.velocity[i] = v_safe[i];
        pub_safe->publish(msg_out);
    }

    void add_constraint(std::vector<Eigen::MatrixXd>& C, std::vector<double>& L, 
                       Eigen::Vector3d p, Eigen::Vector3d obs, double d, double h, 
                       int id, Eigen::Vector3d p_frame) {
        Eigen::Vector3d n;
        if (d > 1e-5) {
            n = (p - obs) / d;
        } else {
            n = (p-p_frame).normalized();
        }
        Eigen::MatrixXd J(6, nv_self); 
        J.setZero(); 
        pinocchio::getFrameJacobian(model_self, data_self, id, pinocchio::LOCAL_WORLD_ALIGNED, J);
        Eigen::MatrixXd J_pt = J.topRows(3) - skew(p - p_frame) * J.bottomRows(3);
        C.push_back(n.transpose() * J_pt); 
        L.push_back(-alpha * h);
    }

    void apply_limits(Eigen::MatrixXd& C, Eigen::VectorXd& L, Eigen::VectorXd& U, 
                 const std::vector<Eigen::MatrixXd>& Cr, const std::vector<double>& Lr, double dt) {
        C.setZero();
        L.setConstant(-1e20);
        U.setConstant(1e20);
        int row_idx = 0;
        
        // Apply Obstacle constraint
        for (size_t i=0; i<Cr.size() && i<30; ++i) { 
            if (row_idx >= C.rows()) break;
            C.row(row_idx) = Cr[i]; 
            L(row_idx) = std::clamp(Lr[i], -5.0, 5.0);
            row_idx++;
        }
        
        // Apply Joint velocity limit
        for (int i=0; i<nv_self; ++i) {
            if (row_idx >= C.rows()) break;
            
            C(row_idx, i) = 1.0;
            
            double limit_l = std::max(-v_limit(i), (q_min(i) - q_safe(i)) / dt);
            double limit_u = std::min(v_limit(i), (q_max(i) - q_safe(i)) / dt);

            if (limit_l > limit_u) {
                double mid = (limit_l + limit_u) / 2.0;
                limit_l = mid - 1e-4;
                limit_u = mid + 1e-4;
            }
            L(row_idx) = limit_l;
            U(row_idx) = limit_u;
            row_idx++;
        }
    }

    Eigen::Vector3d closest_point_on_seg(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d p) {
        Eigen::Vector3d ab = b - a; 
        double len_sq = ab.squaredNorm();
        if (len_sq < 1e-6){
            return a;
        }
        double t = std::clamp((p - a).dot(ab) / ab.squaredNorm(), 0.0, 1.0); 
        return a + t * ab;
    }

    Eigen::Matrix3d skew(Eigen::Vector3d v) { 
        Eigen::Matrix3d m; 
        m << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0; 
        return m; 
    }

    void publish_markers() {
        visualization_msgs::msg::MarkerArray ma;

        std::string root_frame = "base";
        if (capsules_self.find("base") != capsules_self.end()) {
             root_frame = capsules_self["base"].start_frame; 
        }

        int id = 1;
        for(const auto& [name, cap] : capsules_self) {
            if (!model_self.existFrame(cap.start_frame) || !model_self.existFrame(cap.end_frame)) continue;
            
            Eigen::Vector3d p1_raw = data_self.oMf[model_self.getFrameId(cap.start_frame)].translation();
            Eigen::Vector3d p2_raw = data_self.oMf[model_self.getFrameId(cap.end_frame)].translation();

            Eigen::Vector3d p1 = p1_raw;
            Eigen::Vector3d p2 = p2_raw;
            Eigen::Vector3d dir = p2_raw - p1_raw;
            
            if (dir.norm() > 1e-4) {
                dir.normalize();
                double extension = 0.10; 
                p1 = p1_raw - (dir * extension);
                p2 = p2_raw + (dir * extension);
            }
            
            visualization_msgs::msg::Marker m;
            m.header.frame_id = root_frame; 
            m.header.stamp = this->get_clock()->now();
            m.id = id++; 
            m.action = visualization_msgs::msg::Marker::ADD;
            m.color.g = 1.0; 
            m.color.a = 0.4; 

            double len = (p2 - p1).norm();

            if (len < 1e-4) {
                // If start and end are same point, draw a Sphere
                m.type = visualization_msgs::msg::Marker::SPHERE;
                m.pose.position.x = p1.x(); 
                m.pose.position.y = p1.y(); 
                m.pose.position.z = p1.z();
                m.scale.x = m.scale.y = m.scale.z = cap.radius * 2.0;
                
                // Identity orientation
                m.pose.orientation.w = 1.0; 
            } else {
                // Normal Cylinder Capsule
                m.type = visualization_msgs::msg::Marker::CYLINDER;
                m.pose.position.x = (p1.x()+p2.x())/2.0; 
                m.pose.position.y = (p1.y()+p2.y())/2.0; 
                m.pose.position.z = (p1.z()+p2.z())/2.0;
                
                Eigen::Quaterniond q; 
                q.setFromTwoVectors(Eigen::Vector3d::UnitZ(), p2 - p1);
                m.pose.orientation.w = q.w();
                m.pose.orientation.x = q.x(); 
                m.pose.orientation.y = q.y();
                m.pose.orientation.z = q.z();
                
                m.scale.x = m.scale.y = cap.radius * 2.0; 
                m.scale.z = len; 
            }
            ma.markers.push_back(m);
        }
        pub_marker->publish(ma);
    }
};

int main(int argc, char **argv) { 
    rclcpp::init(argc, argv); 
    rclcpp::spin(std::make_shared<SafetyNode>()); 
    rclcpp::shutdown(); 
    return 0; 
}