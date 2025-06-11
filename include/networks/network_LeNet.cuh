#pragma once

#include "network.cuh"

#include "conv2d_forward.cuh"
#include "maxpool2d_forward.cuh"
#include "relu_forward.cuh"
#include "fc_forward.cuh"
#include "flatten_forward.cuh"

class Network_LeNet : public Network
{
public:
    Network_LeNet(float* input, float* output, size_t width, size_t height, cudaStream_t stream);

    void forward() override;

    void loadWeights() override;

    void loadBias() override;

    inline float* getOutput() { return output_; }

private:

    void buildNetwork(float* input, float* output) override;

    float* output_;
    CudaArray<float> output_last_layer_;

    BinaryLoader loader_;
    std::vector<std::unique_ptr<Layer>> layers_;
    cudaStream_t m_stream_;

    struct NetworkWeights
    {
        float* conv1_w;
        float* conv2_w;
        float* fc1_w;
        float* fc2_w;
        float* fc3_w;
    };

    struct NetworkBiases
    {
        float* conv1_b;
        float* conv2_b;
        float* fc1_b;
        float* fc2_b;
        float* fc3_b;
    };

    NetworkWeights weights_;
    NetworkBiases biases_;
};