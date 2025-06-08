#include "conv2d_forward.cuh"

__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* output,
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride, int padding
)
{
    int H_out = (H + 2 * padding - KH) / stride + 1;
    int W_out = (W + 2 * padding - KW) / stride + 1;

    int n = blockIdx.z;
    int co = blockIdx.y;
    int ho = blockIdx.x / W_out;
    int wo = blockIdx.x % W_out;

    if (ho >= H_out || wo >= W_out) return;

    float val = 0.0f;
    for (int ci = 0; ci < C_in; ci++)
    {
        for (int ky = 0; ky < KH; ky++)
        {
            for (int kx = 0; kx < KW; kx++)
            {
                int in_y = ho * stride - padding + ky;
                int in_x = wo * stride - padding + kx;

                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W)
                {
                    int in_index = ((n * C_in + ci) * H + in_y) * W + in_x;
                    int w_index = ((co * C_in + ci) * KH + ky) * KW + kx;
                    val += input[in_index] * weights[w_index];
                }
            }
        }
    }

    if (bias) val += bias[co];

    int out_index = ((n * C_out + co) * H_out + ho) * W_out + wo;
    output[out_index] = val;
}

void conv2d_forward(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride, int padding
)
{
    int H_out = (H + 2 * padding - KH) / stride + 1;
    int W_out = (W + 2 * padding - KW) / stride + 1;

    dim3 blocks(1, 1, 1);
    dim3 grid(W_out * H_out, C_out, N);

    conv2d_forward_kernel<<<grid, blocks>>>(
        input, weights, bias, output,
        N, C_in, H, W,
        C_out, KH, KW,
        stride, padding
    );

    cudaDeviceSynchronize();
}