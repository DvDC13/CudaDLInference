#include "network_LeNet.cuh"

Network_LeNet::Network_LeNet(float* input, float* output, size_t width, size_t height, cudaStream_t stream)
    : output_(output), m_stream_(stream), loader_("../python")
{
    loadWeights();
    loadBias();
    buildNetwork(input, output_);
}

void Network_LeNet::forward()
{
    for (auto& layer : layers_)
    {
        layer->forward(m_stream_);
    }

    output_last_layer_.copyToHost(output_);
}

void Network_LeNet::loadWeights()
{
    loader_.loadInCudaArray<float>("conv1.weight", 6 * 1 * 5 * 5);   
    loader_.loadInCudaArray<float>("conv2.weight", 16 * 6 * 5 * 5);
    loader_.loadInCudaArray<float>("fc1.weight", 120 * 256);
    loader_.loadInCudaArray<float>("fc2.weight", 84 * 120);
    loader_.loadInCudaArray<float>("fc3.weight", 10 * 84);

    weights_.conv1_w = loader_.get<float>("conv1.weight");
    weights_.conv2_w = loader_.get<float>("conv2.weight");
    weights_.fc1_w = loader_.get<float>("fc1.weight");
    weights_.fc2_w = loader_.get<float>("fc2.weight");
    weights_.fc3_w = loader_.get<float>("fc3.weight");
}

void Network_LeNet::loadBias()
{
    loader_.loadInCudaArray<float>("conv1.bias", 6);
    loader_.loadInCudaArray<float>("conv2.bias", 16);
    loader_.loadInCudaArray<float>("fc1.bias", 120);
    loader_.loadInCudaArray<float>("fc2.bias", 84);
    loader_.loadInCudaArray<float>("fc3.bias", 10);

    biases_.conv1_b = loader_.get<float>("conv1.bias");
    biases_.conv2_b = loader_.get<float>("conv2.bias");
    biases_.fc1_b = loader_.get<float>("fc1.bias");
    biases_.fc2_b = loader_.get<float>("fc2.bias");
    biases_.fc3_b = loader_.get<float>("fc3.bias");
}

void Network_LeNet::buildNetwork(float* input, float* output)
{
    CudaArray<float> input_gpu;
    input_gpu.allocate(1, 28, 28);
    input_gpu.copyToDevice(input);

    CudaArray<float> conv1_out_gpu;
    conv1_out_gpu.allocate(1, 6, 24, 24);
    layers_.emplace_back(std::make_unique<Conv2DLayer>(input_gpu.data(), weights_.conv1_w, biases_.conv1_b, conv1_out_gpu.data(), 1, 1, 28, 28, 6, 5, 5, 1, 0));

    layers_.emplace_back(std::make_unique<ReluLayer>(conv1_out_gpu.data(), 1, 6, 24, 24));

    CudaArray<float> pool1_out_gpu;
    pool1_out_gpu.allocate(1, 6, 12, 12);
    layers_.emplace_back(std::make_unique<MaxPool2DLayer>(conv1_out_gpu.data(), pool1_out_gpu.data(), 1, 6, 24, 24, 2, 2, 2));

    CudaArray<float> conv2_out_gpu;
    conv2_out_gpu.allocate(1, 16, 8, 8);
    layers_.emplace_back(std::make_unique<Conv2DLayer>(pool1_out_gpu.data(), weights_.conv2_w, biases_.conv2_b, conv2_out_gpu.data(), 1, 6, 12, 12, 16, 5, 5, 1, 0));

    layers_.emplace_back(std::make_unique<ReluLayer>(conv2_out_gpu.data(), 1, 16, 8, 8));

    CudaArray<float> pool2_out_gpu;
    pool2_out_gpu.allocate(1, 16, 4, 4);
    layers_.emplace_back(std::make_unique<MaxPool2DLayer>(conv2_out_gpu.data(), pool2_out_gpu.data(), 1, 16, 8, 8, 2, 2, 2));

    CudaArray<float> flatten_out_gpu;
    flatten_out_gpu.allocate(1, 256);
    layers_.emplace_back(std::make_unique<FlattenLayer>(pool2_out_gpu.data(), flatten_out_gpu.data(), 1, 16, 4, 4));

    CudaArray<float> fc1_out_gpu;
    fc1_out_gpu.allocate(1, 120);
    layers_.emplace_back(std::make_unique<FCLayer>(flatten_out_gpu.data(), weights_.fc1_w, biases_.fc1_b, fc1_out_gpu.data(), 1, 256, 120));

    layers_.emplace_back(std::make_unique<ReluLayer>(fc1_out_gpu.data(), 1, 120, 1, 1));

    CudaArray<float> fc2_out_gpu;
    fc2_out_gpu.allocate(1, 84);
    layers_.emplace_back(std::make_unique<FCLayer>(fc1_out_gpu.data(), weights_.fc2_w, biases_.fc2_b, fc2_out_gpu.data(), 1, 120, 84));

    layers_.emplace_back(std::make_unique<ReluLayer>(fc2_out_gpu.data(), 1, 84, 1, 1));

    output_last_layer_.allocate(1, 10);
    layers_.emplace_back(std::make_unique<FCLayer>(fc2_out_gpu.data(), weights_.fc3_w, biases_.fc3_b, output_last_layer_.data(), 1, 84, 10));
}