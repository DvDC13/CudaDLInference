#pragma once

void maxpool2d_forward(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pool_H, int pool_W,
    int stride
);