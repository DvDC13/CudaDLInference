#pragma once

#include <vector>
#include <memory>

#include "binary_loader.hxx"
#include "layer.cuh"

class Network
{
public:
    virtual ~Network() = default;
    virtual void forward() = 0;
    virtual void loadWeights() = 0;
    virtual void loadBias() = 0;

private:
    virtual void buildNetwork(float* input, float* output) = 0;
};