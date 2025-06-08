#pragma once

#include <unordered_map>
#include <string>
#include <memory>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <iostream>

#include "cuda_array.cuh"

class BinaryLoader
{
public:
    BinaryLoader(const std::string& path);
    
    // Load and store binary file as a CudaArray<T>
    template<typename T>
    void load(const std::string& name, size_t count);

    // Get device pointer
    template<typename T>
    T* get(const std::string& name);

private:
    std::string m_base_path_;

    struct IArrayHolder
    {
        virtual ~IArrayHolder() = default;
    };

    template<typename T>
    struct ArrayHolder : IArrayHolder
    {
        std::unique_ptr<CudaArray<T>> m_array_;
    };

    std::unordered_map<std::string, std::unique_ptr<IArrayHolder>> m_entries_;
};