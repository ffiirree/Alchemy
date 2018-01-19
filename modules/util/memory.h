#ifndef ALCHEMY_UTIL_MEMORY_H
#define ALCHEMY_UTIL_MEMORY_H

#include <cstddef>

namespace alchemy {

class Memory {
public:
    enum Status {
        UNINITED,  //! 未初始化(未分配内存)
        AT_CPU,    //! 在CPU
        AT_GPU,    //! 在GPU
        SYNCED     //! 数据同步后
    };
public:
    Memory() = default;
    explicit Memory(int size);
    ~Memory();

    inline void * cpu_data() { to_cpu(); return cpu_data_; }
    inline void * gpu_data() { to_gpu(); return gpu_data_; }

    inline size_t size() const { return size_; }

    inline Status status() const { return status_; }

    static void malloc_host(void ** ptr, size_t size);
    static void malloc_device(void ** ptr, size_t size);
    static void free_host(void * ptr);
    static void free_device(void *ptr);

    static void copy(size_t size, void * dst, const void * src);

private:

    void to_cpu();
    void to_gpu();

    Status status_ = UNINITED;
    size_t size_ = 0;

    void *cpu_data_ = nullptr;
    void *gpu_data_ = nullptr;
};

}


#endif //! ALCHEMY_UTIL_MEMORY_H
