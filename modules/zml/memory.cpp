#include <glog/logging.h>
#include <cuda_runtime.h>
#include "memory.hpp"

namespace z {

Memory::Memory(int size)
        : size_(static_cast<size_t>(size))
{}

Memory::~Memory()
{
    if(data_cpu_) free_host(data_cpu_);
    if(data_gpu_) free_device(data_gpu_);
}

void Memory::to_cpu()
{
    switch(status_) {
        case UNINITED:
            malloc_host(&data_cpu_, size_);
            status_ = AT_CPU;
            break;

        case AT_GPU:
            if(!data_cpu_) {
                malloc_host(&data_cpu_, size_);
            }
            copy(size_, data_cpu_, data_gpu_);
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
            malloc_device(&data_cpu_, size_);
            status_ = AT_GPU;
            break;

        case AT_CPU:
            if(!data_gpu_) {
                malloc_device(&data_gpu_, size_);
            }
            copy(size_, data_gpu_, data_cpu_);
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
    cudaMalloc(ptr, size);
}

void Memory::free_device(void *ptr)
{
    cudaFree(ptr);
}

void Memory::copy(size_t count, void *dst, const void *src)
{
    cudaMemcpy(dst, src, count, cudaMemcpyDefault);
}


}
