project(neural-networks)
cmake_minimum_required(VERSION 2.8)

if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
    endif ()
else ()
    set(CMAKE_CXX_STANDARD 11)
endif ()

# Subdirectories
add_subdirectory(mathlib)

# Main
add_executable(main main.cpp)

# Link libraries
target_link_libraries(main mathlib)

target_include_directories(main PUBLIC
    "${PROJECT_BINARY_DIR}"
    "${PROJECT_SOURCE_DIR}/mathlib"
)