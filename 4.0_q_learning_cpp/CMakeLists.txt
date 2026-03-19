cmake_minimum_required(VERSION 3.14.0)

project(qlearning)

# Find protobuf package
find_package(Protobuf REQUIRED)

# Generate C++ files from proto
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS q_learning.proto)

add_executable(qlearning main.cc q_learning.cc ${PROTO_SRCS} ${PROTO_HDRS})
target_include_directories(qlearning PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(qlearning pthread protobuf::libprotobuf)
