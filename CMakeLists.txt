cmake_minimum_required(VERSION 3.21)
project(cudaDLInference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

include_directories(include)

add_executable(${PROJECT_NAME} src/main.cu
    src/conv2d_forward.cu
    src/maxpool2d_forward.cu
    src/relu_forward.cu
    src/fc_forward.cu
    src/flatten_forward.cu
    src/binary_loader.cu
)