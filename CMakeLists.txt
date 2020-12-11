project(neural-networks)
cmake_minimum_required(VERSION 3.6)
set(CMAKE_CXX_STANDARD 14)

# Add Subprojects
add_subdirectory(mathlib)
add_subdirectory(dataload)

# Add Tests
add_subdirectory(mathlib.tests)


# Add Main
add_executable(main main.cpp)
# Link libraries
target_link_libraries(main mathlib)
target_link_libraries(main dataload)
target_include_directories(main PUBLIC
    "${PROJECT_BINARY_DIR}"
    "${PROJECT_SOURCE_DIR}/mathlib"
    "${PROJECT_SOURCE_DIR}/dataload"
)