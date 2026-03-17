cmake_minimum_required(VERSION 3.14.0)

project(bellman_dag)

add_executable(bellman_update_cheese_example ./bellman_update.cc)
target_link_libraries(bellman_update_cheese_example pthread)