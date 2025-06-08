#pragma once

void fc_forward(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int N, int D_in, int D_out
);
