cmake_minimum_required(VERSION 3.20)
project(cbf_safety_layer_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(interactive_markers REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(pinocchio REQUIRED)
find_package(proxsuite REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(interactive_markers REQUIRED)

add_executable(safety_node_cpp src/safety_node.cpp)

# --- FIX PART 1: Use ament_target_dependencies for ROS ---
# This macro automatically finds the .hpp files AND the .so files
ament_target_dependencies(safety_node_cpp
  rclcpp
  sensor_msgs
  visualization_msgs
  geometry_msgs
  interactive_markers
  tf2_ros
  tf2_eigen
  ament_index_cpp
  interactive_markers
)

# --- FIX PART 2: Manual Link for Math (NO "PUBLIC" KEYWORD) ---
# We omit "PUBLIC" so this matches the style used by ament_target_dependencies.
# This prevents the "mixed keyword signature" error.
target_link_libraries(safety_node_cpp
  pinocchio::pinocchio
  proxsuite::proxsuite
)

# Standard Include Setup
target_include_directories(safety_node_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${Eigen3_INCLUDE_DIRS}
)

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