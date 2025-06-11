#include "relu_forward.cuh"

__global__ void relu_forward_kernel(
    float* input,
    int N, int C, int H, int W
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int size = N * C * H * W;

    if (index < size)
    {
        input[index] = fmaxf(0.0f, input[index]);
    }
}