cmake_minimum_required(VERSION 3.14.0)

project(qlearning)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} src_files)

add_executable(qlearning ${src_files})
target_link_libraries(qlearning pthread)