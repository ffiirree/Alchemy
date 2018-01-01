#ifndef _ZML_MEMORY_HPP
#define _ZML_MEMORY_HPP

#include <cstddef>

namespace z {

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

    inline void * data_cpu() const { return data_cpu_; }
    inline void * data_gpu() const { return data_gpu_; }

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

    void *data_cpu_ = nullptr;
    void *data_gpu_ = nullptr;
};

}


#endif //!_ZML_MEMORY_HPP
