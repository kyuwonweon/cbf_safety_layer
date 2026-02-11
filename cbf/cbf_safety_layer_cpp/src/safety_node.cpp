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

using namespace std::chrono_literals;

struct Capsule {
    std::string start_frame;
    std::string end_frame;
    double radius;
};

class SafetyNode : public rclcpp::Node {
public:
    SafetyNode() : Node("safety_node_cpp") {
        // load robot
        this->declare_parameter("robot_description", rclcpp::ParameterValue(std::string("")));
        std::string urdf_string;
        rclcpp::sleep_for(std::chrono::milliseconds(500)); 
        
        if (!this->get_parameter("robot_description", urdf_string) || urdf_string.empty()) {
            RCLCPP_WARN(this->get_logger(), "Param 'robot_description' empty. Using fallback.");
            std::string urdf_pkg = ament_index_cpp::get_package_share_directory("franka_description");
            std::string urdf_path = urdf_pkg + "/robots/fer/fer.urdf";
            pinocchio::urdf::buildModel(urdf_path, model_);
        } else {
            pinocchio::urdf::buildModelFromXML(urdf_string, model_);
        }
        
        data_ = pinocchio::Data(model_);
        nq_ = model_.nq;
        nv_ = model_.nv;
        RCLCPP_INFO(this->get_logger(), "Model Loaded: nq=%d, nv=%d", nq_, nv_);

        q_min_ = Eigen::VectorXd(nv_);
        q_max_ = Eigen::VectorXd(nv_);
        v_limit_ = Eigen::VectorXd(nv_);

        // Standard Franka Research 3 Limits (Radians & Rad/s)
        q_min_ << -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0;
        q_max_ <<  2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973, 0.04, 0.04;
        v_limit_ << 2.1750,  2.1750,  2.1750,  2.1750,  2.6100,  2.6100,  2.6100, 0.1, 0.1;

        // Safety Buffer to prevent hitting hard-stops
        double padding = 0.02; 
        q_min_ = q_min_.array() + padding;
        q_max_ = q_max_.array() - padding;
        v_limit_ *= 0.95;

        q_safe_ = Eigen::VectorXd::Zero(nq_);
        v_safe_ = Eigen::VectorXd::Zero(nv_);

        // Capsule
        capsules_["base"]      = {"fer_link0", "fer_link1", 0.15};
        capsules_["shoulder"]  = {"fer_link1", "fer_link3", 0.10};
        capsules_["upper_arm"] = {"fer_link3", "fer_link4", 0.10};
        capsules_["forearm"]   = {"fer_link4", "fer_link7", 0.13};
        capsules_["hand"]      = {"fer_link7", "fer_hand",  0.11};

        v_user_command_ = Eigen::VectorXd::Zero(nv_);

        // ROS Setup
        auto qos = rclcpp::QoS(10);
        sub_js_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states_source", qos, std::bind(&SafetyNode::joint_cb, this, std::placeholders::_1));
        pub_safe_ = this->create_publisher<sensor_msgs::msg::JointState>("/safety/joint_states", qos);
        pub_marker_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/safety_marker", qos);

        pub_cmd_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/velocity_group_controller/commands", 10);

        // Update this block in Constructor
        sub_input_vel_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/safety/input_joint_states", 10, 
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
                // Fix: Cast nv_ to size_t
                if (msg->velocity.size() == (size_t)nv_) {
                    for(int i=0; i<nv_; ++i) v_user_command_(i) = msg->velocity[i];

                    if (v_user_command_.norm() > 0.1) {
                        RCLCPP_INFO(this->get_logger(), "RECV CMD: Joint1=%.2f", v_user_command_(0));
                    
                    }
                }
            });
        
        // Interactive Marker
        server_ = std::make_shared<interactive_markers::InteractiveMarkerServer>("obstacle_server", this);
        create_interactive_obstacle();
        last_viz_time_ = this->get_clock()->now();
        RCLCPP_INFO(this->get_logger(), "Safety Node Ready");
    }

private:
    pinocchio::Model model_;
    pinocchio::Data data_;
    int nq_, nv_;
    int loop_count_ = 0;
    rclcpp::Time last_viz_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    Eigen::VectorXd q_safe_, v_safe_;
    Eigen::Vector3d obs_pose_ = {0.8, 0.0, 0.4}; 
    double obs_radius_ = 0.10;
    double safety_margin_ = 0.02;
    double alpha_ = 5.0;

    Eigen::VectorXd v_user_command_; 
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_input_vel_;
    
    std::map<std::string, Capsule> capsules_;
    
    // Global solver pointer
    std::shared_ptr<proxsuite::proxqp::dense::QP<double>> qp_;
    
    bool first_run_ = true;
    bool solver_initialized_ = false;
    rclcpp::Time last_time_;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_js_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_safe_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_marker_;
    rclcpp::TimerBase::SharedPtr viz_timer_;
    std::shared_ptr<interactive_markers::InteractiveMarkerServer> server_;
    Eigen::VectorXd q_min_, q_max_, v_limit_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_input_v_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_cmd_;

    void create_interactive_obstacle() {
        visualization_msgs::msg::InteractiveMarker int_marker;
        int_marker.header.frame_id = "base";
        int_marker.name = "obstacle_marker";
        int_marker.scale = 0.3;
        int_marker.pose.position.x = obs_pose_(0);
        int_marker.pose.position.y = obs_pose_(1);
        int_marker.pose.position.z = obs_pose_(2);

        visualization_msgs::msg::Marker box_marker;
        box_marker.type = visualization_msgs::msg::Marker::SPHERE;
        box_marker.scale.x = obs_radius_ * 2.0;
        box_marker.scale.y = obs_radius_ * 2.0;
        box_marker.scale.z = obs_radius_ * 2.0;
        box_marker.color.r = 1.0; box_marker.color.a = 1.0;

        visualization_msgs::msg::InteractiveMarkerControl box_control;
        box_control.always_visible = true;
        box_control.markers.push_back(box_marker);
        int_marker.controls.push_back(box_control);

        visualization_msgs::msg::InteractiveMarkerControl control;

        // X Axis (Red Arrow)
        control.orientation.w = 1;
        control.orientation.x = 1;
        control.orientation.y = 0;
        control.orientation.z = 0;
        control.name = "move_x";
        control.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS;
        int_marker.controls.push_back(control);

        // Z Axis (Blue Arrow)
        control.orientation.w = 1;
        control.orientation.x = 0;
        control.orientation.y = 1;
        control.orientation.z = 0;
        control.name = "move_z";
        control.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS;
        int_marker.controls.push_back(control);

        // Y Axis (Green Arrow)
        control.orientation.w = 1;
        control.orientation.x = 0;
        control.orientation.y = 0;
        control.orientation.z = 1;
        control.name = "move_y";
        control.interaction_mode = visualization_msgs::msg::InteractiveMarkerControl::MOVE_AXIS;
        int_marker.controls.push_back(control);

        server_->insert(int_marker);
        server_->setCallback(int_marker.name, std::bind(&SafetyNode::process_int_marker, this, std::placeholders::_1));
        server_->applyChanges();
    }

    void process_int_marker(const visualization_msgs::msg::InteractiveMarkerFeedback::ConstSharedPtr & feedback) {
        if (feedback->event_type == visualization_msgs::msg::InteractiveMarkerFeedback::POSE_UPDATE) {
            obs_pose_ << feedback->pose.position.x, feedback->pose.position.y, feedback->pose.position.z;
        }
    }

    void joint_cb(const sensor_msgs::msg::JointState::SharedPtr msg) {
        auto start_time = std::chrono::high_resolution_clock::now();
        auto now = this->get_clock()->now();
        if (first_run_) {
            last_time_ = now;
            last_viz_time_ = now;
            first_run_ = false;
        }
        double dt = (now - last_time_).seconds();
        last_time_ = now;
        if (dt < 0.001) return;

        Eigen::VectorXd q_input = Eigen::VectorXd::Zero(nq_);
        size_t limit = std::min((size_t)nq_, msg->position.size());
        for(size_t i=0; i < limit; ++i){
            q_input[i] = msg->position[i];
        }

        q_safe_ = q_input;
        Eigen::VectorXd v_des = v_user_command_;
        v_des = v_des.cwiseMin(2.0).cwiseMax(-2.0);

        // Joint Limit Buffers
        double buffer = 0.15; 
        for (int i = 0; i < nv_; ++i) {
            double dist_upper = q_max_(i) - q_safe_(i);
            if (dist_upper < buffer && v_des(i) > 0) v_des(i) *= std::max(0.0, dist_upper / buffer);
            double dist_lower = q_safe_(i) - q_min_(i);
            if (dist_lower < buffer && v_des(i) < 0) v_des(i) *= std::max(0.0, dist_lower / buffer);
        }

        // Kinematics Update
        pinocchio::forwardKinematics(model_, data_, q_safe_);
        pinocchio::updateFramePlacements(model_, data_);
        pinocchio::computeJointJacobians(model_, data_, q_safe_);

        // Constraint Generation
        std::vector<Eigen::MatrixXd> C_rows;
        std::vector<double> l_vals, u_vals;

        for (const auto& [name, capsule] : capsules_) {
            if (!model_.existFrame(capsule.end_frame) || !model_.existFrame(capsule.start_frame)) continue;
            auto id_start = model_.getFrameId(capsule.start_frame);
            auto id_end = model_.getFrameId(capsule.end_frame);
            
            // Raw frame positions for Jacobian calc
            Eigen::Vector3d p_frame_start = data_.oMf[id_start].translation(); 
            Eigen::Vector3d p_frame_end = data_.oMf[id_end].translation();

            Eigen::Vector3d p_s = p_frame_start;
            Eigen::Vector3d p_e = p_frame_end;
            
            Eigen::Vector3d axis = p_e - p_s;
            double len = axis.norm();
            if (len > 1e-5) {
                Eigen::Vector3d dir = axis.normalized();
                if (name != "base") {
                    p_s -= dir * 0.10;
                }
                if (capsule.end_frame != "fer_hand_tcp"){
                    p_e += dir * 0.10;
                }
            }

            // Floor check (using extended points)
            if (name != "base") {
                double h_floor = p_e[2] - capsule.radius - safety_margin_;
                if (h_floor < 0.2) { 
                    Eigen::MatrixXd J_end(6, nv_); J_end.setZero();
                    pinocchio::getFrameJacobian(model_, data_, id_end, pinocchio::LOCAL_WORLD_ALIGNED, J_end);
                    // Adjust Jacobian for lever arm from Frame to Extended Tip
                    Eigen::Vector3d lever = p_e - p_frame_end;
                    Eigen::MatrixXd J_tip = J_end.topRows(3) - skew(lever) * J_end.bottomRows(3);

                    C_rows.push_back(J_tip.row(2)); 
                    l_vals.push_back(-alpha_ * std::max(h_floor, 0.001)); 
                    u_vals.push_back(1e20); 
                }
            }

            auto res = closest_segment_point(p_s, p_e, obs_pose_);
            double dist = (res.first - res.second).norm();
            double h_obs = dist - (capsule.radius + 0.05 + safety_margin_);
            if (h_obs < 0.4) {
                Eigen::Vector3d n = (dist > 1e-6) ? ((res.first - res.second) / dist) : Eigen::Vector3d(1,0,0);
                Eigen::MatrixXd J_start(6, nv_); J_start.setZero();
                pinocchio::getFrameJacobian(model_, data_, id_start, pinocchio::LOCAL_WORLD_ALIGNED, J_start);
                Eigen::Vector3d lever = res.first - p_frame_start;
                Eigen::MatrixXd J_point = J_start.topRows(3) - skew(lever) * J_start.bottomRows(3);
                C_rows.push_back(n.transpose() * J_point);
                l_vals.push_back(-alpha_ * h_obs);
                u_vals.push_back(1e20);
            }
        }

        // QP solver
        const int max_constraints = 40; 
        Eigen::MatrixXd C_total = Eigen::MatrixXd::Zero(max_constraints, nv_);
        Eigen::VectorXd l_total = Eigen::VectorXd::Constant(max_constraints, -1e20);
        Eigen::VectorXd u_total = Eigen::VectorXd::Constant(max_constraints, +1e20);

        applyJointLimits(C_total, l_total, u_total, C_rows, l_vals, u_vals, dt);

        Eigen::MatrixXd H = Eigen::MatrixXd::Identity(nv_, nv_);
        Eigen::VectorXd g = -v_des;
        Eigen::MatrixXd A_eq(0, nv_); Eigen::VectorXd b_eq(0);

        try {
            if (!solver_initialized_) {
                qp_ = std::make_shared<proxsuite::proxqp::dense::QP<double>>(nv_, 0, max_constraints);
                qp_->settings.eps_abs = 1.0e-5; qp_->settings.verbose = false;
                qp_->init(H, g, A_eq, b_eq, C_total, l_total, u_total);
                solver_initialized_ = true;
            } else {
                qp_->update(H, g, A_eq, b_eq, C_total, l_total, u_total);
            }
            qp_->solve();
            if (qp_->results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
                v_safe_ = qp_->results.x;
            } else {
                v_safe_.setZero();
            }
        } catch (const std::exception& e) {
            v_safe_.setZero();
            solver_initialized_ = false; 
        }

        if (loop_count_ % 50 == 0 && v_des.norm() > 0.1) {
             RCLCPP_INFO(this->get_logger(), "QP OUTPUT: v_safe=[%.2f, %.2f...]", v_safe_(0), v_safe_(1));
        }

        std_msgs::msg::Float64MultiArray cmd_msg;
        for(int i=0; i<7; ++i) cmd_msg.data.push_back(v_safe_[i]); 
        pub_cmd_->publish(cmd_msg);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        if (duration>800){
            RCLCPP_WARN(this -> get_logger(), "Safety Loop slow: %ld us",duration);
        }

        q_safe_ += v_safe_ * dt;

        sensor_msgs::msg::JointState msg_out;
        msg_out.header.stamp = this->get_clock()->now();
        msg_out.header.frame_id = "base";
        msg_out.name = {
            "fer_joint1", "fer_joint2", "fer_joint3", "fer_joint4", 
            "fer_joint5", "fer_joint6", "fer_joint7", 
            "fer_finger_joint1", "fer_finger_joint2"
        };

        msg_out.position.resize(9);
        msg_out.velocity.resize(9);
        for(int i=0; i<9; ++i) {
            msg_out.position[i] = q_safe_[i];
            msg_out.velocity[i] = v_safe_[i];
        }
        pub_safe_->publish(msg_out);

        loop_count_++;
        if ((now - last_viz_time_).seconds() > 0.1) {
            publish_markers();
            last_viz_time_ = now;
        }
    }

    void publish_markers() {
        visualization_msgs::msg::MarkerArray ma;

        visualization_msgs::msg::Marker delete_all;
        delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
        delete_all.id = 0;
        ma.markers.push_back(delete_all);

        int id = 1;
        for(const auto& [name, capsule] : capsules_) {
            if (!model_.existFrame(capsule.end_frame) || !model_.existFrame(capsule.start_frame)) continue;

            auto id_start = model_.getFrameId(capsule.start_frame);
            auto id_end = model_.getFrameId(capsule.end_frame);
            
            Eigen::Vector3d p1 = data_.oMf[id_start].translation();
            Eigen::Vector3d p2 = data_.oMf[id_end].translation();

            Eigen::Vector3d axis = p2 - p1;
            if (axis.norm() > 1e-5) {
                Eigen::Vector3d dir = axis.normalized();
                if (name != "base") p1 -= dir * 0.10; 
                p2 += dir * 0.10;
            }

            visualization_msgs::msg::Marker m;
            
            m.header.frame_id = "base";
            m.header.stamp = rclcpp::Time(0);
            m.id = id++;
            m.type = visualization_msgs::msg::Marker::CYLINDER;
            m.action = visualization_msgs::msg::Marker::ADD;
            
            Eigen::Vector3d center = (p1 + p2) / 2.0;
            Eigen::Vector3d diff = p2 - p1;
            
            double len = diff.norm();
            m.pose.position.x = center.x();
            m.pose.position.y = center.y();
            m.pose.position.z = center.z();
            
            Eigen::Vector3d z_axis(0,0,1);
            Eigen::Quaterniond q; 
            q.setFromTwoVectors(z_axis, diff);
            
            m.pose.orientation.w = q.w();
            m.pose.orientation.x = q.x();
            m.pose.orientation.y = q.y();
            m.pose.orientation.z = q.z();
            m.scale.x = capsule.radius * 2.0;
            m.scale.y = capsule.radius * 2.0;
            m.scale.z = len;
            m.color.r = 0.0;
            m.color.g = 1.0;
            m.color.b = 0.0;
            m.color.a = 0.5;
            m.lifetime = rclcpp::Duration::from_seconds(0);
            ma.markers.push_back(m);
        }
        pub_marker_->publish(ma);
    }

    void applyJointLimits(
        Eigen::MatrixXd& C_total, 
        Eigen::VectorXd& l_total, 
        Eigen::VectorXd& u_total,
        const std::vector<Eigen::MatrixXd>& C_obs,
        const std::vector<double>& l_obs,
        const std::vector<double>& u_obs,
        double dt) 
    {
        int num_obs = C_obs.size();
        int max_rows = C_total.rows();
        
        if (num_obs + nv_ > max_rows) {
            RCLCPP_ERROR(this->get_logger(), "Too many constraints! Increase max_constraints in joint_cb.");
            return; 
        }

        // Reset the matrix to Zero to clear old constraints
        C_total.setZero();
        
        // Reset bounds to infinity to have the robot unconstrained by default
        l_total.setConstant(-1e20);
        u_total.setConstant(1e20);

        // Obstacle constraint
        for (int i = 0; i < num_obs; ++i) {
            C_total.row(i) = C_obs[i];
            l_total(i) = l_obs[i];
            u_total(i) = u_obs[i];
        }

        // Joint and velocity limits
        for (int i = 0; i < nv_; ++i) {
            int row = num_obs + i;
            C_total(row, i) = 1.0;
            double v_to_min = (q_min_(i) - q_safe_(i)) / dt;
            double v_to_max = (q_max_(i) - q_safe_(i)) / dt;
            v_to_min = std::clamp(v_to_min, -v_limit_(i), v_limit_(i));
            v_to_max = std::clamp(v_to_max, -v_limit_(i), v_limit_(i));
            l_total(row) = std::max(-v_limit_(i), v_to_min);
            u_total(row) = std::min(v_limit_(i), v_to_max);
        }
    }

    Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
        Eigen::Matrix3d m;
        m << 0, -v(2), v(1),
             v(2), 0, -v(0),
             -v(1), v(0), 0;
        return m;
    }

    std::pair<Eigen::Vector3d, Eigen::Vector3d> closest_segment_point(
        Eigen::Vector3d p1, Eigen::Vector3d p2, Eigen::Vector3d p_obs) 
    {
        Eigen::Vector3d v1 = p2 - p1;
        Eigen::Vector3d gap = p1 - p_obs;
        double len_sq = v1.squaredNorm();
        double t = 0.0;
        if (len_sq > 1e-6) t = std::clamp(-gap.dot(v1) / len_sq, 0.0, 1.0);
        return {p1 + t * v1, p_obs};
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SafetyNode>());
    rclcpp::shutdown();
    return 0;
}