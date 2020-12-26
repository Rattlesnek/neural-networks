project(NNLib)
cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 17)

# Find multiprocessing library OpenMP
find_package(OpenMP)

include_directories(${CMAKE_SOURCE_DIR}/mathlib)

link_directories(${CMAKE_BINARY_DIR}/mathlib)

file(GLOB_RECURSE CPP_FILES "*.cpp")
file(GLOB_RECURSE H_FILES "*.h")
file(GLOB_RECURSE HPP_FILES "*.hpp")

add_library(${PROJECT_NAME} SHARED
	${CPP_FILES}
)

add_dependencies(${PROJECT_NAME} mathlib)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall ${OpenMP_CXX_FLAGS})

target_link_libraries(${PROJECT_NAME} mathlib ${OpenMP_CXX_FLAGS})
