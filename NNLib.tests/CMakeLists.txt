project(NNLib.tests)
cmake_minimum_required(VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

enable_testing()

# Include directory with header files of the library to be tested
include_directories(${CMAKE_SOURCE_DIR}/NNLib)
include_directories(${CMAKE_SOURCE_DIR}/mathlib)

# Link directory with library to be tested
link_directories(${CMAKE_BINARY_DIR}/NNLib)
link_directories(${CMAKE_BINARY_DIR}/mathlib)

file(GLOB_RECURSE CPP_FILES "*.cpp")
file(GLOB_RECURSE H_FILES "*.h")
file(GLOB_RECURSE HPP_FILES "*.hpp")

# Create tests executable
add_executable(${PROJECT_NAME}
	${CPP_FILES}
)

# Add dependencies and link libraries
add_dependencies(${PROJECT_NAME} NNLib)
add_dependencies(${PROJECT_NAME} mathlib)


target_link_libraries(${PROJECT_NAME} NNLib mathlib gtest gtest_main pthread) 

# Add the test
add_test(NAME ${PROJECT_NAME}
	COMMAND ${PROJECT_NAME}
)