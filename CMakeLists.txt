cmake_minimum_required(VERSION 3.22)
project(my_liblinear)

set(CMAKE_CXX_STANDARD 14)

add_executable(my_fm main.cpp linear.h linear.cpp)
