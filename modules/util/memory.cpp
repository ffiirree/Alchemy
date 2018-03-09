#include "memory.h"

#include <glog/logging.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace alchemy {

Memory::Memory(int size)
        : size_(static_cast<size_t>(size))
{}

Memory::~Memory()
{
    if(cptr_) {
        free_host(cptr_);
        cptr_ = nullptr;
    };
    if(gptr_) {
        free_device(gptr_);
        gptr_ = nullptr;
    }
}

void Memory::to_cpu()
{
    switch(status_) {
        case UNINITED:
            malloc_host(&cptr_, size_);
            status_ = AT_CPU;
            break;

        case AT_GPU:
            if(!cptr_) {
                malloc_host(&cptr_, size_);
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
            malloc_device(&gptr_, size_);
            status_ = AT_GPU;
            break;

        case AT_CPU:
            if(!gptr_) {
                malloc_device(&gptr_, size_);
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

void Memory::malloc_host(void ** ptr, size_t size)
{
#ifdef USE_CUDA
    cudaMallocHost(ptr, size);
#else
    *ptr = malloc(size);
#endif
}

void Memory::free_host(void *ptr)
{
#ifdef USE_CUDA
    cudaFreeHost(ptr);
#else
    free(ptr);
#endif
}

void Memory::malloc_device(void ** ptr, size_t size)
{
#ifdef USE_CUDA
    cudaMalloc(ptr, size);
#else
    LOG(FATAL) << "NO GPU!";
#endif
}

void Memory::free_device(void *ptr)
{
#ifdef USE_CUDA
    cudaFree(ptr);
#else
    LOG(FATAL) << "NO GPU!";
#endif
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
