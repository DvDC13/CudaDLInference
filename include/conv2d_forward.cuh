#pragma once

void conv2d_forward(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride, int padding
);