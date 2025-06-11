#pragma once

#include "layer.cuh"

__global__ void fc_forward_kernel(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int N, int D_in, int D_out
);

class FCLayer : public Layer
{
public:
    FCLayer(float* input, float* weights, float* bias, float* output, int N, int D_in, int D_out)
        : input_(input), output_(output), weights_(weights), bias_(bias), in_N_(N), D_in_(D_in), D_out_(D_out) {}

    void forward(cudaStream_t stream) override
    {
        launchKernel(fc_forward_kernel, dim3(D_out_, 1, in_N_), dim3(1, 1, 1), stream,
            input_, weights_, bias_, output_, in_N_, D_in_, D_out_);
    }

private:
    float* input_;
    float* output_;
    float* weights_;
    float* bias_;
    int in_N_;
    int D_in_;
    int D_out_;
};
