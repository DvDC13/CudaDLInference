cmake_minimum_required(VERSION 3.21)
project(cudaDLInference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

include_directories(include)
include_directories(include/layers)
include_directories(include/networks)

add_executable(${PROJECT_NAME} src/main.cu
    src/layers/conv2d_forward.cu
    src/layers/maxpool2d_forward.cu
    src/layers/relu_forward.cu
    src/layers/fc_forward.cu
    src/layers/flatten_forward.cu
    src/binary_loader.cu
    src/networks/network_LeNet.cu
)