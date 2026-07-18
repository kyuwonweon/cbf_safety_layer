cmake_minimum_required(VERSION 3.20)
project(cbf_safety_layer_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(franka_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(interactive_markers REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(pinocchio REQUIRED)
find_package(proxsuite REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Franka 0.10.0 REQUIRED)

# =========================================================
# BUILD RULES FOR SAFETY_NODE_CPP
# =========================================================
add_executable(safety_node_cpp src/safety_node.cpp)

target_link_libraries(safety_node_cpp PUBLIC
  ${geometry_msgs_TARGETS}
  ${sensor_msgs_TARGETS}
  ${franka_msgs_TARGETS}
  ${std_msgs_TARGETS}
  ${std_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
  ament_index_cpp::ament_index_cpp
  interactive_markers::interactive_markers
  rclcpp::rclcpp
  sensor_msgs::sensor_msgs_library
  tf2_eigen::tf2_eigen
  tf2_ros::static_transform_broadcaster_node
  tf2_ros::tf2_ros
  Franka::Franka
  pinocchio::pinocchio
  proxsuite::proxsuite
)

target_include_directories(safety_node_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Eigen3_INCLUDE_DIRS}
)

# =========================================================
# INSTALLATION
# =========================================================
install(TARGETS safety_node_cpp
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS src/test.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()