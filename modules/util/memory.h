#ifndef ALCHEMY_UTIL_MEMORY_H
#define ALCHEMY_UTIL_MEMORY_H

#include <cstddef>
#include <cuda_runtime.h>
#include "util.h"

namespace alchemy {
template <typename Device> void alchemy_alloc(void **ptr, size_t size);
template <typename Device> void alchemy_free(void *ptr);

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
    explicit Memory(size_t size);
    ~Memory();

    inline const void * cptr() { to_cpu(); return (const void *)cptr_; }
    inline const void * gptr() { to_gpu(); return (const void *)gptr_; }
    
    inline void * mutable_cptr() { to_cpu(); status_ = AT_CPU; return cptr_; }
    inline void * mutable_gptr() { to_gpu(); status_ = AT_GPU; return gptr_; }
    
    inline size_t size() const { return size_; }

    inline Status status() const { return status_; }

    static void copy(size_t size, void * dst, const void * src);

private:
    void to_cpu();
    void to_gpu();

    Status status_ = UNINITED;
    size_t size_ = 0;

    void *cptr_ = nullptr;
    void *gptr_ = nullptr;
};
}
#endif //! ALCHEMY_UTIL_MEMORY_H
