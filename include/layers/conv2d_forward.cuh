#pragma once

#include "layer.cuh"

__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* output,
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride, int padding
);

class Conv2DLayer : public Layer
{
public:
    Conv2DLayer(float* input, float* weights, float* bias, float* output,
                int in_N, int in_C, int in_H, int in_W, int out_C, int KH, int KW, int stride, int padding)
        : input_(input), output_(output), weights_(weights), bias_(bias),
          in_N_(in_N), in_C_(in_C), in_H_(in_H), in_W_(in_W), out_C_(out_C), KH_(KH), KW_(KW), stride_(stride), padding_(padding) {}

    void forward(cudaStream_t stream) override
    {
        int H_out = (in_H_ + 2 * padding_ - KH_) / stride_ + 1;
        int W_out = (in_W_ + 2 * padding_ - KW_) / stride_ + 1;

        launchKernel(conv2d_forward_kernel, dim3(W_out * H_out, out_C_, in_N_), dim3(1, 1, 1), stream,
            input_, weights_, bias_, output_, in_N_, in_C_, in_H_, in_W_, out_C_, KH_, KW_, stride_, padding_);
    }

private:
    float* input_, * output_, * weights_, * bias_;
    int in_N_,in_C_, in_H_, in_W_, out_C_, KH_, KW_, stride_, padding_;
};
