#pragma once

#include "layer.cuh"

__global__ void maxpool2d_forward_kernel(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pool_H, int pool_W,
    int stride
);

class MaxPool2DLayer : public Layer
{
public:
    MaxPool2DLayer(float* input, float* output, int N, int C, int H, int W, int pool_H, int pool_W, int stride)
        : input_(input), output_(output), in_N_(N), in_C_(C), in_H_(H), in_W_(W), pool_H_(pool_H), pool_W_(pool_W), stride_(stride) {}

    void forward(cudaStream_t stream) override
    {
        const int H_out = (in_H_ - pool_H_) / stride_ + 1;
        const int W_out = (in_W_ - pool_W_) / stride_ + 1;

        launchKernel(maxpool2d_forward_kernel, dim3(H_out * W_out, in_C_, in_N_), dim3(1, 1, 1), stream,
            input_, output_, in_N_, in_C_, in_H_, in_W_, pool_H_, pool_W_, stride_);
    }

private:
    float* input_;
    float* output_;
    int in_N_;
    int in_C_;
    int in_H_;
    int in_W_;
    int pool_H_;
    int pool_W_;
    int stride_;
};