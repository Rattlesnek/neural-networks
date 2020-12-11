project(dataload.tests)
cmake_minimum_required(VERSION 3.6)
set(CMAKE_CXX_STANDARD 14)

enable_testing()

# Include directory with header files of the library to be tested
include_directories(${CMAKE_SOURCE_DIR}/dataload)

# Link directory with library to be tested
link_directories(${CMAKE_BINARY_DIR}/dataload)

file(GLOB_RECURSE CPP_FILES "*.cpp")
file(GLOB_RECURSE H_FILES "*.h")
file(GLOB_RECURSE HPP_FILES "*.hpp")

# Create tests executable
add_executable(${PROJECT_NAME}
	${CPP_FILES}
)

# Add dependencies and link libraries
add_dependencies(${PROJECT_NAME} dataload)

target_link_libraries(${PROJECT_NAME} dataload gtest gtest_main pthread) 

# Add the test
add_test(NAME ${PROJECT_NAME}
	COMMAND ${PROJECT_NAME}
)