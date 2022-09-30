cmake_minimum_required(VERSION 3.0.2)
project(grasp_detection_msgs)


set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")


find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_generation
  roscpp
  rospy
  sensor_msgs
  std_msgs
)


## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Grasp.msg
)


## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs#   std_msgs
)


catkin_package(
 CATKIN_DEPENDS cv_bridge image_transport message_generation roscpp rospy sensor_msgs std_msgs
)
