#include <iostream>
#include <algorithm>

#include "network_LeNet.cuh"

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
    size_t N = 10000, C = 1, H = 28, W = 28;

    BinaryLoader loader("../python");

    std::vector<float> testInput = loader.loadImages<float>("../python/mnist_images.bin", N, C, H, W);
    std::vector<float> testLabels = loader.loadLabels<float>("../python/mnist_labels.bin", N);
    std::vector<float> expected_output = loader.loadInVector<float>("mnist_output", N * 10);

    for (int i = 0; i < 1; i++)
    {
        float* inputImage_cpu = testInput.data() + i * C * H * W;

        float* output_cpu = new float[10];
        Network_LeNet network(inputImage_cpu, output_cpu, H, W, 0);
        network.forward();

        if (compareOutputs(output_cpu, expected_output.data() + i * 10, 10)) {
            std::cout << "✅ Output matches expected_output.bin\n";
        } else {
            std::cerr << "❌ Mismatch detected.\n";
        }

        int predicted = std::distance(output_cpu, std::max_element(output_cpu, output_cpu + 10));

        int ground_truth = testLabels[i];
        
        if (predicted != ground_truth) {
            std::cout << "Wrong prediction at " << i << ": predicted " << predicted << ", actual " << ground_truth << '\n';
        }
    }

    return EXIT_SUCCESS;
}