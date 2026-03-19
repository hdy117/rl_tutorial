cmake_minimum_required(VERSION 3.14.0)

project(qlearning)

add_executable(qlearning main.cc q_learning.cc)
target_link_libraries(qlearning pthread)