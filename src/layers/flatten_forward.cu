#include "flatten_forward.cuh"

__global__ void flatten_forward_kernel(
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