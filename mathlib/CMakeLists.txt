project(mathlib)
cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE CPP_FILES "*.cpp")
file(GLOB_RECURSE H_FILES "*.h")
file(GLOB_RECURSE HPP_FILES "*.hpp")

add_library(${PROJECT_NAME} SHARED
	${CPP_FILES}
)
