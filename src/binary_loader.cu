#include "binary_loader.hxx"

BinaryLoader::BinaryLoader(const std::string& path) : m_base_path_(path) {}

template<typename T>
void BinaryLoader::load(const std::string& name, size_t count)
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

    std::cout << "✅ Loaded " << name << std::endl;

    file.close();
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

template void BinaryLoader::load<float>(const std::string& name, size_t count);
template float* BinaryLoader::get<float>(const std::string& name);