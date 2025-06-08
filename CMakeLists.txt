cmake_minimum_required(VERSION 3.21)
project(cudaDLInference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

add_executable(${PROJECT_NAME} src/main.cu)