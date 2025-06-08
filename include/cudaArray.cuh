#pragma once

#include "error.cuh"

template<typename T>
__host__ __device__ constexpr auto getSize(T v) { return v; }

template<typename T, typename... Tail>
__host__ __device__ constexpr auto getSize(T v, Tail... tail) { return v * getSize(tail...); }

template<typename T>
class CudaArray
{
public:
    CudaArray() : m_data(nullptr), m_size(0) {}

    ~CudaArray()
    {
        if (m_data)
            checkCudaErrors(cudaFree(m_data));
    }

    template<typename... Dims>
    void allocate(Dims... dims)
    {
        m_size = getSize(static_cast<size_t>(dims)...);
        checkCudaErrors(cudaMalloc(&m_data, sizeBytes()));
    }

    void copyToDevice(T* data)
    {
        checkCudaErrors(cudaMemcpy(m_data, data, m_size, cudaMemcpyHostToDevice));
    }

    void copyToHost(T* data)
    {
        checkCudaErrors(cudaMemcpy(data, m_data, m_size, cudaMemcpyDeviceToHost));
    }

    inline T* data() const { return m_data; }
    inline size_t size() const { return m_size; }
    inline size_t sizeBytes() const { return m_size * sizeof(T); }

private:
    T* m_data;
    size_t m_size;
};