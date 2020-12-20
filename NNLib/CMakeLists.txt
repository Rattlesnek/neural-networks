project(NNLib)
cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/mathlib)

link_directories(${CMAKE_BINARY_DIR}/mathlib)

file(GLOB_RECURSE CPP_FILES "*.cpp")
file(GLOB_RECURSE H_FILES "*.h")
file(GLOB_RECURSE HPP_FILES "*.hpp")

add_library(${PROJECT_NAME} SHARED
	${CPP_FILES}
)

add_dependencies(${PROJECT_NAME} mathlib)

target_link_libraries(${PROJECT_NAME} mathlib)
