cmake_minimum_required(VERSION 3.8)
project(elf3_arm_ik_interfaces)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SolveArmIK.srv"
  "action/PlanArmTrajectory.action"
  DEPENDENCIES geometry_msgs sensor_msgs trajectory_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
