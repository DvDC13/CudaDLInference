#include <iostream>

#include "conv2d_forward.cuh"
#include "maxpool2d_forward.cuh"
#include "relu_forward.cuh"
#include "fc_forward.cuh"
#include "flatten_forward.cuh"

#include "binary_loader.hxx"

bool compareOutputs(const float* output1, const float* output2, size_t size, float tol=1e-5f) {
    for (size_t i = 0; i < size; ++i) {
        if (std::fabs(output1[i] - output2[i]) > tol) {
            std::cerr << "Mismatch at " << i << ": " << output1[i] << " vs " << output2[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(void)
{
    BinaryLoader loader("../python");

    loader.loadInCudaArray<float>("conv1.weight", 6 * 1 * 5 * 5);
    loader.loadInCudaArray<float>("conv1.bias", 6);
    loader.loadInCudaArray<float>("conv2.weight", 16 * 6 * 5 * 5);
    loader.loadInCudaArray<float>("conv2.bias", 16);
    loader.loadInCudaArray<float>("fc1.weight", 120 * 256);
    loader.loadInCudaArray<float>("fc1.bias", 120);
    loader.loadInCudaArray<float>("fc2.weight", 84 * 120);
    loader.loadInCudaArray<float>("fc2.bias", 84);
    loader.loadInCudaArray<float>("fc3.weight", 10 * 84);
    loader.loadInCudaArray<float>("fc3.bias", 10);

    auto* conv1_w = loader.get<float>("conv1.weight");
    auto* conv1_b = loader.get<float>("conv1.bias");
    auto* conv2_w = loader.get<float>("conv2.weight");
    auto* conv2_b = loader.get<float>("conv2.bias");
    auto* fc1_w = loader.get<float>("fc1.weight");
    auto* fc1_b = loader.get<float>("fc1.bias");
    auto* fc2_w = loader.get<float>("fc2.weight");
    auto* fc2_b = loader.get<float>("fc2.bias");
    auto* fc3_w = loader.get<float>("fc3.weight");
    auto* fc3_b = loader.get<float>("fc3.bias");

    std::cout << "✅ Loaded weights" << std::endl;

    std::vector<float> inputImage_cpu = loader.loadInVector<float>("test_input", 28 * 28);

    CudaArray<float> input_gpu;
    input_gpu.allocate(1, 28, 28);
    input_gpu.copyToDevice(inputImage_cpu.data());

    CudaArray<float> conv1_out_gpu;
    conv1_out_gpu.allocate(1, 6, 24, 24);
    conv2d_forward(input_gpu.data(), conv1_w, conv1_b, conv1_out_gpu.data(), 1, 1, 28, 28, 6, 5, 5, 1, 0);

    relu_forward(conv1_out_gpu.data(), 1, 6, 24, 24);

    CudaArray<float> pool1_out_gpu;
    pool1_out_gpu.allocate(1, 6, 12, 12);
    maxpool2d_forward(conv1_out_gpu.data(), pool1_out_gpu.data(), 1, 6, 24, 24, 2, 2, 2);

    CudaArray<float> conv2_out_gpu;
    conv2_out_gpu.allocate(1, 16, 8, 8);
    conv2d_forward(pool1_out_gpu.data(), conv2_w, conv2_b, conv2_out_gpu.data(), 1, 6, 12, 12, 16, 5, 5, 1, 0);

    relu_forward(conv2_out_gpu.data(), 1, 16, 8, 8);

    CudaArray<float> pool2_out_gpu;
    pool2_out_gpu.allocate(1, 16, 4, 4);
    maxpool2d_forward(conv2_out_gpu.data(), pool2_out_gpu.data(), 1, 16, 8, 8, 2, 2, 2);

    CudaArray<float> flatten_out_gpu;
    flatten_out_gpu.allocate(1, 256);
    flatten_forward(pool2_out_gpu.data(), flatten_out_gpu.data(), 1, 16, 4, 4);

    CudaArray<float> fc1_out_gpu;
    fc1_out_gpu.allocate(1, 120);
    fc_forward(flatten_out_gpu.data(), fc1_w, fc1_b, fc1_out_gpu.data(), 1, 256, 120);

    relu_forward(fc1_out_gpu.data(), 1, 120, 1, 1);

    CudaArray<float> fc2_out_gpu;
    fc2_out_gpu.allocate(1, 84);
    fc_forward(fc1_out_gpu.data(), fc2_w, fc2_b, fc2_out_gpu.data(), 1, 120, 84);

    relu_forward(fc2_out_gpu.data(), 1, 84, 1, 1);

    CudaArray<float> fc3_out_gpu;
    fc3_out_gpu.allocate(1, 10);
    fc_forward(fc2_out_gpu.data(), fc3_w, fc3_b, fc3_out_gpu.data(), 1, 84, 10);

    float output_cpu[10];
    fc3_out_gpu.copyToHost(output_cpu);

    std::vector<float> expected_output = loader.loadInVector<float>("expected_output", 10);

    if (compareOutputs(output_cpu, expected_output.data(), 10)) {
        std::cout << "✅ Output matches expected_output.bin\n";
    } else {
        std::cerr << "❌ Mismatch detected.\n";
        return 1;
    }

    return EXIT_SUCCESS;
}