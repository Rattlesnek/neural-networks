project(mathlib)
cmake_minimum_required(VERSION 2.8)

if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
    endif ()
else ()
    set(CMAKE_CXX_STANDARD 11)
endif ()

file(GLOB_RECURSE CPP_FILES "*.cpp")
file(GLOB_RECURSE H_FILES "*.h")
file(GLOB_RECURSE HPP_FILES "*.hpp")

add_library(${PROJECT_NAME} SHARED
	${CPP_FILES}
)
