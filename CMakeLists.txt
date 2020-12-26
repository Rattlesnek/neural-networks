project(neural-networks)
cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 17)

# Find multiprocessing library OpenMP
find_package(OpenMP REQUIRED)

# Add Subprojects
add_subdirectory(mathlib)
add_subdirectory(NNLib)
add_subdirectory(dataload)

# Add Tests
add_subdirectory(mathlib.tests)
add_subdirectory(NNLib.tests)
add_subdirectory(dataload.tests)

# Add Main
add_executable(main main.cpp)
# Compile options
target_compile_options(main PRIVATE -Wall ${OpenMP_CXX_FLAGS})
# Link libraries
target_link_libraries(main mathlib NNLib dataload ${OpenMP_CXX_FLAGS})

target_include_directories(main PUBLIC
    "${PROJECT_BINARY_DIR}"
    "${PROJECT_SOURCE_DIR}/mathlib"
    "${PROJECT_SOURCE_DIR}/NNLib"
    "${PROJECT_SOURCE_DIR}/dataload"
)

#Add second main
add_executable(2main 2main.cpp)

target_link_libraries(2main mathlib NNLib dataload)

target_include_directories(2main PUBLIC
    "${PROJECT_BINARY_DIR}"
    "${PROJECT_SOURCE_DIR}/mathlib"
    "${PROJECT_SOURCE_DIR}/NNLib"
    "${PROJECT_SOURCE_DIR}/dataload"
)
