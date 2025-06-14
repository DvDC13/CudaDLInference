#include "fc_forward.cuh"

__global__ void fc_forward_kernel(
    const float* input,     // [N, D_in]
    const float* weights,   // [D_out, D_in]
    const float* bias,      // [D_out] or nullptr
    float* output,          // [N, D_out]
    int N, int D_in, int D_out
)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int size = N * D_out;

    if (index < size)
    {
        int n = index / D_out;
        int d_out = index % D_out;

        float sum = 0.0f;
        for (int d_in = 0; d_in < D_in; d_in++)
        {
            sum += input[n * D_in + d_in] * weights[d_out * D_in + d_in];
        }
        if (bias != nullptr)
        {
            sum += bias[d_out];
        }
        output[n * D_out + d_out] = sum;
    }
}