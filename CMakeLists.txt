cmake_minimum_required(VERSION 3.5)
project(NN)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
add_executable(NN main.cpp NN.cpp NN.h node.h)