#pragma once

#include <assert.h>

class Layer {
public:
    virtual ~Layer() = default;
    virtual void forward(cudaStream_t stream) = 0;

    template<typename Kernel, typename... Args>
    void launchKernel(Kernel kernel, dim3 grid, dim3 block, cudaStream_t stream, Args&&... args)
    {
        assert(grid.x > 0 && grid.y > 0 && grid.z > 0);
        assert(block.x > 0 && block.y > 0 && block.z > 0);
        
        kernel<<<grid, block, 0, stream>>>(std::forward<Args>(args)...);
    }
};