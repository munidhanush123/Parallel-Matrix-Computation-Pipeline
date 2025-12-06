cmake_minimum_required(VERSION 3.12)
project(ParallelTaskflowMatrix LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Include Taskflow
include_directories(deps/taskflow)

# Include headers (if any future additions)
include_directories(include)

# Source files
add_executable(taskflow_matrix
    src/main.cpp
)

# Required threading
target_link_libraries(taskflow_matrix pthread)

