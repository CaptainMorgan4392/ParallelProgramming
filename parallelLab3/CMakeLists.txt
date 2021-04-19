cmake_minimum_required(VERSION 3.19)
project(parallelLab3 C)

set(CMAKE_C_STANDARD 99)

find_package(MPI REQUIRED)
include_directories(MPI_INCLUDE_DIRS)

add_executable(parallelLab3 main.c)
target_link_libraries(parallelLab3 MPI::MPI_C)