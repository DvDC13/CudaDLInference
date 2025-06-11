#pragma once

#include "layer.cuh"

__global__ void flatten_forward_kernel(
    const float* input,
    float* output,
    int N, int C, int H, int W
);

class FlattenLayer : public Layer
{
public:
    FlattenLayer(float* input, float* output, int N, int C, int H, int W)
        : input_(input), output_(output), in_N_(N), in_C_(C), in_H_(H), in_W_(W) {}

    void forward(cudaStream_t stream) override
    {
        int size = in_N_ * in_C_ * in_H_ * in_W_;

        launchKernel(flatten_forward_kernel, dim3((size + 255) / 256, 1, 1), dim3(256, 1, 1), stream,
            input_, output_, in_N_, in_C_, in_H_, in_W_);
    }

private:
    float* input_;
    float* output_;
    int in_N_;
    int in_C_;
    int in_H_;
    int in_W_;
};