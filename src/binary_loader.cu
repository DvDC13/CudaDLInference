#include "binary_loader.hxx"

BinaryLoader::BinaryLoader(const std::string& path) : m_base_path_(path) {}

template<typename T>
void BinaryLoader::loadInCudaArray(const std::string& name, size_t count)
{
    std::string fullPath = m_base_path_ + '/' + name + ".bin";
    
    std::ifstream file(fullPath, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + fullPath);
    }

    std::vector<T> hostData(count);
    file.read(reinterpret_cast<char*>(hostData.data()), count * sizeof(T));
    if (!file)
    {
        throw std::runtime_error("Unable to read file: " + fullPath);
    }

    auto holder = std::make_unique<ArrayHolder<T>>();
    holder->m_array_ = std::make_unique<CudaArray<T>>();
    holder->m_array_->allocate(count);
    holder->m_array_->copyToDevice(hostData.data());

    m_entries_[name] = std::move(holder);

    std::cout << "✅ Loaded: " << name << " (" << count << " " << typeid(T).name() << ")" << std::endl;

    file.close();
}

template<typename T>
std::vector<T> BinaryLoader::loadInVector(const std::string& name, size_t count)
{
    std::ifstream file(m_base_path_ + '/' + name + ".bin", std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + m_base_path_ + '/' + name + ".bin");
    }

    size_t size = file.tellg();
    if (size != count * sizeof(T))
    {
        throw std::runtime_error("Invalid file size: " + m_base_path_ + '/' + name + ".bin");
    }

    file.seekg(0, std::ios::beg);

    std::vector<T> hostData(count);
    file.read(reinterpret_cast<char*>(hostData.data()), size);
    if (!file)
    {
        throw std::runtime_error("Unable to read file: " + m_base_path_ + '/' + name + ".bin");
    }

    return hostData;
}

template<typename T>
std::vector<T> BinaryLoader::loadImages(const std::string& filename, size_t N, size_t C, size_t H, size_t W) {
    size_t total = N * C * H * W;
    std::vector<T> data(total);

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open image file: " + filename);
    }

    file.read(reinterpret_cast<char*>(data.data()), total * sizeof(T));
    if (!file) {
        throw std::runtime_error("Failed to read image data from: " + filename);
    }

    return data;
}

template<typename T>
std::vector<T> BinaryLoader::loadLabels(const std::string& filename, size_t N) {
    std::vector<T> labels(N);

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open label file: " + filename);
    }

    file.read(reinterpret_cast<char*>(labels.data()), N * sizeof(T));
    if (!file) {
        throw std::runtime_error("Failed to read label data from: " + filename);
    }

    return labels;
}

template<typename T>
T* BinaryLoader::get(const std::string& name)
{
    auto it = m_entries_.find(name);
    if (it == m_entries_.end())
    {
        throw std::runtime_error("Entry not found: " + name);
    }
    
    auto* holder = dynamic_cast<ArrayHolder<T>*>(it->second.get());
    if (!holder)
    {
        throw std::runtime_error("Invalid entry type: " + name);
    }

    return holder->m_array_->data();
}

template void BinaryLoader::loadInCudaArray<float>(const std::string& name, size_t count);
template std::vector<float> BinaryLoader::loadInVector<float>(const std::string& name, size_t count);
template std::vector<float> BinaryLoader::loadImages<float>(const std::string& name, size_t N, size_t C, size_t H, size_t W);
template std::vector<float> BinaryLoader::loadLabels<float>(const std::string& filename, size_t N);
template float* BinaryLoader::get<float>(const std::string& name);