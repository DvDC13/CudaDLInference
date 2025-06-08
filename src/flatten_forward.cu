#include "flatten_forward.cuh"

__global__ void flatten_kernel(
    const float* input,
    float* output,
    int N, int C, int H, int W
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int size = N * C * H * W;

    if (index < size)
    {
        output[index] = input[index];
    }
}

void flatten(
    const float* input,
    float* output,
    int N, int C, int H, int W
)
{
    dim3 blocks(1, 1, 1);
    dim3 grid(W * H * C * N, 1, 1);

    flatten_kernel<<<grid, blocks>>>(
        input, output,
        N, C, H, W
    );
}