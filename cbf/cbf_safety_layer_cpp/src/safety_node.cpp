#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <interactive_markers/interactive_marker_server.hpp> 
#include <ament_index_cpp/get_package_share_directory.hpp>

// Math Libraries
#include <Eigen/Dense>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <proxsuite/proxqp/dense/dense.hpp> // Using ProxSuite
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

        q_safe_ = Eigen::VectorXd::Zero(nq_);
        v_safe_ = Eigen::VectorXd::Zero(nv_);

        // Capsule
        capsules_["base"]      = {"fer_link0", "fer_link1", 0.13};
        capsules_["upper_arm"] = {"fer_link1", "fer_link4", 0.12};
        capsules_["forearm"]   = {"fer_link4", "fer_link6", 0.10};
        capsules_["hand"]      = {"fer_link6", "fer_hand",  0.10};
        capsules_["finger_L"]  = {"fer_hand",  "fer_leftfinger", 0.05};
        capsules_["finger_R"]  = {"fer_hand",  "fer_rightfinger", 0.04};

        // ROS Setup
        auto qos = rclcpp::QoS(10);
        sub_js_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", qos, std::bind(&SafetyNode::joint_cb, this, std::placeholders::_1));
        pub_safe_ = this->create_publisher<sensor_msgs::msg::JointState>("/safety/joint_states", qos);
        pub_marker_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/safety_marker", qos);
        viz_timer_ = this->create_wall_timer(100ms, std::bind(&SafetyNode::viz_cb, this));

        // Interactive Marker
        server_ = std::make_shared<interactive_markers::InteractiveMarkerServer>("obstacle_server", this);
        create_interactive_obstacle();
        
        RCLCPP_INFO(this->get_logger(), "Safety Node Ready");
    }

private:
    pinocchio::Model model_;
    pinocchio::Data data_;
    int nq_, nv_;
    
    Eigen::VectorXd q_safe_, v_safe_;
    Eigen::Vector3d obs_pose_ = {0.8, 0.0, 0.4}; 
    double obs_radius_ = 0.10;
    double safety_margin_ = 0.02;
    double alpha_ = 5.0;
    
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
        auto now = this->get_clock()->now();
        if (first_run_) {
            last_time_ = now;
            first_run_ = false;
        }
        double dt = (now - last_time_).seconds();
        last_time_ = now;
        if (dt < 0.001) return;

        Eigen::VectorXd q_input = Eigen::VectorXd::Zero(nq_);
        Eigen::VectorXd v_input = Eigen::VectorXd::Zero(nv_);
        size_t limit = std::min((size_t)nq_, msg->position.size());
        for(size_t i=0; i < limit; ++i){
            q_input[i] = msg->position[i];
        }
        if(msg->velocity.size() > 0) {
             size_t v_limit = std::min((size_t)nv_, msg->velocity.size());
             for(size_t i=0; i < v_limit; ++i) v_input[i] = msg->velocity[i];
        }

        if (q_safe_.isZero(1e-6)){
            q_safe_ = q_input;
        }
        // Control 
        double Kp = 2.0;
        Eigen::VectorXd v_des = v_input + Kp * (q_input - q_safe_);
        v_des = v_des.cwiseMin(2.0).cwiseMax(-2.0);

        // Kinematics 
        pinocchio::forwardKinematics(model_, data_, q_safe_);
        pinocchio::updateFramePlacements(model_, data_);
        pinocchio::computeJointJacobians(model_, data_, q_safe_);

        // Constraints 
        std::vector<Eigen::MatrixXd> C_rows; // Store rows as 1xN matrices
        std::vector<double> l_vals;
        std::vector<double> u_vals;

        for (const auto& [name, capsule] : capsules_) {
            if (!model_.existFrame(capsule.end_frame) || !model_.existFrame(capsule.start_frame)) continue;
            
            auto id_start = model_.getFrameId(capsule.start_frame);
            auto id_end = model_.getFrameId(capsule.end_frame);
            Eigen::Vector3d p_start = data_.oMf[id_start].translation();
            Eigen::Vector3d p_end = data_.oMf[id_end].translation();

            // Floor (Skip Base)
            if (name != "base") {
                double h_floor = p_end[2] - capsule.radius - safety_margin_;
                if (h_floor < 0.2) { 
                    Eigen::MatrixXd J_end(6, nv_);
                    J_end.setZero();
                    pinocchio::getFrameJacobian(model_, data_, id_end, pinocchio::LOCAL_WORLD_ALIGNED, J_end);
                    
                    C_rows.push_back(J_end.row(2)); 
                    l_vals.push_back(-alpha_ * std::max(h_floor, 0.001)); // Prevent NaN
                    u_vals.push_back(1e20); 
                }
            }

            // Obstacle
            auto res = closest_segment_point(p_start, p_end, obs_pose_);
            double dist = (res.first - res.second).norm();
            double calc_radius = 0.05; 
            double h_obs = dist - (capsule.radius + calc_radius + safety_margin_);
            
            if (h_obs < 0.4) {
                Eigen::Vector3d n = (dist > 1e-6) ? ((res.first - res.second) / dist) : Eigen::Vector3d(1,0,0);
                Eigen::MatrixXd J_start(6, nv_);
                J_start.setZero();
                pinocchio::getFrameJacobian(model_, data_, id_start, pinocchio::LOCAL_WORLD_ALIGNED, J_start);
                Eigen::Vector3d lever = res.first - p_start;
                Eigen::MatrixXd J_point = J_start.topRows(3) - skew(lever) * J_start.bottomRows(3);
                
                C_rows.push_back(n.transpose() * J_point);
                l_vals.push_back(-alpha_ * std::max(h_obs, 0.001));
                u_vals.push_back(1e20);
            }
        }

        // Don't go through solver logic if no constraints
        if (C_rows.empty()) {
            v_safe_ = v_des;
        } else {
            // extract constraints from vectors to matrices
            int n_active = C_rows.size();
            Eigen::MatrixXd C_active(n_active, nv_);
            Eigen::VectorXd l_active(n_active);
            Eigen::VectorXd u_active(n_active);

            for(int i=0; i<n_active; ++i) {
                C_active.row(i) = C_rows[i];
                l_active(i) = l_vals[i];
                u_active(i) = u_vals[i];
            }

            Eigen::MatrixXd H = Eigen::MatrixXd::Identity(nv_, nv_);
            Eigen::VectorXd g = -v_des;
            Eigen::MatrixXd A_eq(0, nv_); 
            Eigen::VectorXd b_eq(0);

            try {
                // Check if we need to re-initialize (Dimension change or First Run)
                if (!solver_initialized_ || !qp_ || qp_->model.n_in != n_active) {
                    qp_ = std::make_shared<proxsuite::proxqp::dense::QP<double>>(nv_, 0, n_active);
                    qp_->settings.eps_abs = 1.0e-5;
                    qp_->settings.verbose = false;
                    qp_->init(H, g, A_eq, b_eq, C_active, l_active, u_active);
                    solver_initialized_ = true;
                } else {
                    qp_->update(H, g, A_eq, b_eq, C_active, l_active, u_active);
                }

                qp_->solve();

                if (qp_->results.info.status == proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) {
                    v_safe_ = qp_->results.x;
                } else {
                    v_safe_.setZero(); // Infeasible
                }
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "QP Error: %s", e.what());
                v_safe_.setZero();
                solver_initialized_ = false; // Force re-init next time
            }
        }

        q_safe_ += v_safe_ * dt;

        sensor_msgs::msg::JointState msg_out = *msg;
        if (msg_out.velocity.size() != msg_out.position.size()) {
            msg_out.velocity.resize(msg_out.position.size(), 0.0);
        }
        size_t out_limit = std::min((size_t)nq_, msg_out.position.size());
        for(size_t i=0; i < out_limit; ++i) {
            msg_out.position[i] = q_safe_[i];
            msg_out.velocity[i] = v_safe_[i];
        }
        pub_safe_->publish(msg_out);
    }

    void viz_cb() {
        visualization_msgs::msg::MarkerArray ma;
        int id = 0;
        for(const auto& [name, capsule] : capsules_) {
            if (!model_.existFrame(capsule.end_frame) || !model_.existFrame(capsule.start_frame)) continue;

            auto id_start = model_.getFrameId(capsule.start_frame);
            auto id_end = model_.getFrameId(capsule.end_frame);
            
            Eigen::Vector3d p1 = data_.oMf[id_start].translation();
            Eigen::Vector3d p2 = data_.oMf[id_end].translation();
            visualization_msgs::msg::Marker m;
            
            m.header.frame_id = "base";
            m.header.stamp = this->get_clock()->now();
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
            Eigen::Quaterniond q; q.setFromTwoVectors(z_axis, diff);
            
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
            ma.markers.push_back(m);
        }
        pub_marker_->publish(ma);
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