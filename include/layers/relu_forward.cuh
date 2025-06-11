#pragma once

#include "layer.cuh"

__global__ void relu_forward_kernel(
    float* input,
    int N, int C, int H, int W
);

class ReluLayer : public Layer
{
public:
    ReluLayer(float* input, int N, int C, int H, int W)
        : input_(input), in_N_(N), in_C_(C), in_H_(H), in_W_(W) {}

    void forward(cudaStream_t stream) override
    {
        int size = in_N_ * in_C_ * in_H_ * in_W_;
        dim3 block(256, 1, 1);
        dim3 grid((size + 255) / 256, 1, 1);
        launchKernel(relu_forward_kernel, grid, block, stream, input_, in_N_, in_C_, in_H_, in_W_);
    }

private:
    float* input_;
    int in_N_;
    int in_C_;
    int in_H_;
    int in_W_;
};