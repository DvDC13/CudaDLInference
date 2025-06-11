#include "maxpool2d_forward.cuh"

__global__ void maxpool2d_forward_kernel(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pool_H, int pool_W,
    int stride
)
{
    int H_out = (H - pool_H) / stride + 1;
    int W_out = (W - pool_W) / stride + 1;

    int n = blockIdx.z;
    int c = blockIdx.y;
    int ho = blockIdx.x / W_out;
    int wo = blockIdx.x % W_out;

    if (ho >= H_out || wo >= W_out) return;

    float val = 0.0f;
    for (int ky = 0; ky < pool_H; ky++)
    {
        for (int kx = 0; kx < pool_W; kx++)
        {
            int in_y = ho * stride + ky;
            int in_x = wo * stride + kx;

            if (in_y < H && in_x < W)
            {
                val = fmaxf(val, input[n * C * H * W + c * H * W + in_y * W + in_x]);
            }
        }
    }

    output[n * C * H_out * W_out + c * H_out * W_out + ho * W_out + wo] = val;
}