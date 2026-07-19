# CBF Safety Layer for Franka Emika (CBF-QP)

[![Portfolio](https://img.shields.io/badge/Read_The_Full_Portfolio_Post-blue?style=for-the-badge&logo=ReadTheDocs)](https://kyuwonweon.github.io/projects/cbf_qp/)

## Project Demonstration
*Watch the collision avoidance system in action below:*
[![Watch the video](https://img.youtube.com/vi/68CW9JTyOSY/maxresdefault.jpg)](https://youtu.be/68CW9JTyOSY)
*(Dual Arm Safety Simulation & Real-time Hardware Intervention)*

## Overview
This package implements a **Real-Time Dynamic Collision Prevention System** for the 7-DOF Franka Emika Panda. Using Control Barrier Functions (CBF), this node acts as a transparent safety filter. It intercepts incoming joint velocity or twist commands from a human operator or global planner, and modifies them via a Quadratic Program (QP) solver running strictly at **1kHz**. 

This system mathematically guarantees the robot remains within a safe set of states, preventing self-collisions, floor collisions, and dynamic obstacle collisions, regardless of input aggression.

## Technical Stack
- **Middleware:** ROS 2 Kilted
- **Hardware Interface:** `franka_bringup` / `franka_description`
- **Kinematics:** Pinocchio (Rigid body dynamics and Jacobians)
- **Optimization:** ProxQP (Dense QP Solver)
- **Math:** Eigen3
- **Languages:** C++ (Safety Node / QP Solver), Python (Teleop / Launch)

---

## System Architecture

The system operates as a strict middleware layer, resolving constraints in real-time at 1kHz. 

### Core Node: `/safety_node_cpp`

**Inputs (Intention & State):**
* `/joint_states_source` & `/joint_states_source_other`: 1kHz hardware states for the primary and secondary (dual-arm) robots.
* `/teleop_vel` & `/manual_vel`: The desired user or global planner joint velocity commands.
* `/shared_obstacle`: Dynamic obstacle poses streaming from interactive markers or perception nodes.

**Internal Processing Pipeline:**
1. **Kinematic Update:** Uses **Pinocchio** to compute forward kinematics, frame placements, joint Jacobians, and the Mass Matrix ($M(q)$) via the Composite Rigid Body Algorithm (CRBA).
2. **Geometric Evaluation:** Models the robot using 8 overlapping capsules (from `link0` to `hand`). Calculates closest point-to-segment and segment-to-segment distances algebraically to completely bypass the latency of heavy mesh-collision libraries.
3. **Constraint Formulation:** Constructs the dense matrices ($C$, $l$, $u$) for four distinct hazard types: Floor, Dynamic Obstacles, Inter-Robot collisions, and Self-Collisions.
4. **Optimization:** Solves the constrained Quadratic Program using **ProxQP**.

**Outputs (Action):**
* `/velocity_group_controller/commands`: The verified, mathematically safe velocity commands sent to the hardware.
* `/safety_marker`: RViz2 visualization of the active safety capsules and collision vectors.

### Key Features & Engineering Highlights
* **Minimal Intervention (Zero Local Minima):** Unlike Artificial Potential Fields (APF) which constantly push the robot and cause jitter, the CBF-QP is strictly minimally invasive. If the user's command is safe, the QP passes it through 100% unaltered. It only intervenes at the exact mathematical boundary of danger.
* **Kinetic Energy Awareness:** The dynamic barrier function accounts for the robot's current joint velocities. If the robot is moving rapidly, the safety layer engages earlier to dissipate momentum. If moving slowly, it allows the robot to get much closer to the obstacle safely.
* **1kHz Real-Time Determinism:** Built for strict timing. By pre-allocating all Eigen matrices, avoiding `std::vector` dynamic resizing in the fast path, and capping the ProxQP iterations, the optimization resolves in under 0.5 milliseconds, guaranteeing stable 1000 Hz control loops.
* **Algebraic Collision Primitives:** By abstracting the 7-DOF mesh into bounded capsules and spheres, the system completely bypasses the computational lag of heavy collision engines (like FCL or Bullet), computing analytical distances and gradients instantly.

---

## Mathematics of the Energy-Aware CBF-QP

Unlike standard kinematic safety filters, this implementation treats safety as a dynamic constraint. It accounts for the robot's physical inertia and kinetic energy to guarantee it always has the stopping power required to avoid an impact.

### 1. The Energy-Aware Barrier Function
Let $h(q)$ be the geometric distance between the robot capsule and a hazard. We define a dynamic barrier function $B(q, \dot{q})$ that couples this distance to the robot's kinetic energy ($E_k$):

$$E_k = \frac{1}{2}\dot{q}^T M(q) \dot{q}$$

$$B(q, \dot{q}) = \gamma \max(h(q), -0.05) - E_k$$

To ensure the robot does not enter an unavoidable collision state, we enforce the derivative inequality $\dot{B} \ge -\alpha B$, which bounds the allowable kinetic energy relative to the distance to the obstacle.

### 2. High-Order Kinematic Constraints
Because the robot is commanded at the acceleration level internally, we apply a PD-style High-Order CBF (HOCBF) to constrain the approach rate. Using the geometric Jacobian $J(q)$, the distance constraint is bounded by:

$$l_{kin} = -k_p h(q) - k_d \dot{h}(q)$$

$$l_{kin} \le J(q) \ddot{q} \le \infty$$

*(Note on Dual-Arm Setup: For inter-robot collisions, the Jacobian constraint is mapped backward through a relative transformation matrix $R_{self}^T R_{other}$ to accurately project the secondary robot's velocity into the primary robot's frame).*

### 3. The Dense Quadratic Program (QP)
The solver maps the user's desired velocity into a desired acceleration $\ddot{q}_{des} = k_p(\dot{q}_{des} - \dot{q}_{safe})$ and finds the optimal safe acceleration ($\ddot{q}^*$) using ProxQP. 

$$\min_{\ddot{q}} \frac{1}{2} || \ddot{q} - \ddot{q}_{des} ||^2$$

**Subject to:**

$$
\begin{aligned}
L_{cbf} \le C_{cbf} \ddot{q} \le U_{cbf} &\quad \text{(Energy and Geometry Bounds)} \\
a_{min} \le \ddot{q} \le a_{max} &\quad \text{(Hardware Acceleration Limits)} \\
\dot{q}_{min} \le \dot{q} \le \dot{q}_{max} &\quad \text{(Hardware Velocity Limits)} \\
q_{min} \le q \le q_{max} &\quad \text{(Hardware Joint Limits)}
\end{aligned}
$$

The optimal acceleration is integrated numerically ($v_{safe} = v_{safe} + \ddot{q}^* \Delta t$) and clamped to hardware limits before publishing.
---

## Reproduction Instructions

### 1. Prerequisites
Ensure you have a working ROS 2 Kilted environment and the necessary Franka packages installed. You will also need to install the mathematical dependencies:
```bash
sudo apt install ros-kilted-franka-description ros-kilted-franka-bringup
# Install Pinocchio and ProxQP (Refer to their respective official installation guides)
```

### 2. Build the Workspace
```bash
cd ~/<your_workspace>
colcon build --packages-select cbf_safety_layer_cpp cbf_safety_layer franka_description
source install/setup.bash
```

### 3. Run the Simulation

**Single arm:**
```bash
ros2 launch cbf_safety_layer_cpp mock.launch.py
```
Brings up `robot_state_publisher`, the safety node in simulation mode, a joint-state mover, and RViz (with the safety capsules, the draggable obstacle marker, and TF pre-loaded). To actually drive the arm and see the CBF constraints engage, publish a `sensor_msgs/JointState` with a `velocity` array to `/manual_vel`.

**Dual arm:**
```bash
ros2 launch cbf_safety_layer dual_demo.launch.py
```
Launches two namespaced safety nodes (`/robot1`, `/robot2`) with mirrored bases 1.2m apart, each aware of the other for inter-robot collision avoidance, plus a joystick teleop node per robot and RViz. Requires a joystick connected for `joy_node` to drive them; alternatively publish directly to `/robot1/manual_vel` and `/robot2/manual_vel`.
