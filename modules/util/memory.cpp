#include <glog/logging.h>
#include "memory.h"

namespace alchemy {


template<> void alchemy_alloc<CPU>(void **ptr, size_t size)
{
#ifdef USE_CUDA
    cudaMallocHost(ptr, size);
#else
    *ptr = malloc(size);
#endif
}
template<> void alchemy_alloc<GPU>(void **ptr, size_t size)
{
#ifdef USE_CUDA
    cudaMallocManaged(ptr, size);
#else
    LOG(FATAL) << "NO GPU!";
#endif
}

template<> void alchemy_free<CPU>(void *ptr)
{
#ifdef USE_CUDA
    cudaFreeHost(ptr);
#else
    free(ptr);
#endif
}

template<> void alchemy_free<GPU>(void *ptr)
{
#ifdef USE_CUDA
    cudaFree(ptr);
#else
    LOG(FATAL) << "NO GPU!";
#endif
}
void alchemy_copy(void *dst, void *src, size_t size)
{
#ifdef USE_CUDA
    cudaMemcpy(dst, src, size, cudaMemcpyDefault);
#else
    LOG(FATAL) << "NO GPU!";
#endif
}

Memory::Memory(size_t size)
        : size_(size)
{}

Memory::~Memory()
{
    if(cptr_) {
        alchemy_free<CPU>(cptr_);
        cptr_ = nullptr;
    };
    if(gptr_) {
        alchemy_free<GPU>(gptr_);
        gptr_ = nullptr;
    }
}

void Memory::to_cpu()
{
    switch(status_) {
        case UNINITED:
            alchemy_alloc<CPU>(&cptr_, size_);
            status_ = AT_CPU;
            break;

        case AT_GPU:
            if(!cptr_) {
                alchemy_alloc<CPU>(&cptr_, size_);
            }
            copy(size_, cptr_, gptr_);
            status_ = SYNCED;
            break;

        case AT_CPU:
        case SYNCED:
            break;

        default:
            LOG(INFO) << "Unknown Memory status.";
            break;
    }
}

void Memory::to_gpu()
{
    switch(status_) {
        case UNINITED:
            alchemy_alloc<GPU>(&gptr_, size_);
            status_ = AT_GPU;
            break;

        case AT_CPU:
            if(!gptr_) {
                alchemy_alloc<GPU>(&gptr_, size_);
            }
            copy(size_, gptr_, cptr_);
            status_ = SYNCED;
            break;

        case AT_GPU:
        case SYNCED:
            break;

        default:
            LOG(INFO) << "Unknown Memory status.";
            break;
    }
}

void Memory::copy(size_t count, void *dst, const void *src)
{
#ifdef USE_CUDA
    cudaMemcpy(dst, src, count, cudaMemcpyDefault);
#else
    LOG(FATAL) << "NO GPU!";
#endif
}

}
