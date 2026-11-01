cmake_minimum_required(VERSION 3.0.2)
project(final_project)

find_package(catkin REQUIRED)
catkin_package()

catkin_install_python(PROGRAMS src/pose_imitation.py src/real_robot_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
