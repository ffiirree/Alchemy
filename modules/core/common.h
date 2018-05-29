#ifndef ALCHEMY_CORE_COMMON_H
#define ALCHEMY_CORE_COMMON_H

#include <cassert>
#include <cstddef>
#include <vector>
#include <memory>
#include <map>

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifdef USE_CUDNN
#include <cudnn_v7.h>
#endif
#endif

using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::weak_ptr;
using std::pair;
using std::string;
using std::map;
using std::tuple;

namespace alchemy {

enum Phase {
    TRAIN, TEST, SHARED
};

class Global {
public:
    Global(const Global&) = delete;
    Global&operator=(const Global&) = delete;
    ~Global();

    enum Mode{ CPU, GPU };
    static Mode mode() { return mode_; }
    static void mode(Mode m) { mode_ = m; }

    static Global& Instance();
#ifdef USE_CUDA
    static cublasHandle_t cublas_handle() { return Instance().cublas_handle_; }
#endif
private:
    Global();

    static Global* instance_;
    static Mode mode_;

#ifdef USE_CUDA
    cublasHandle_t cublas_handle_ = nullptr;
#endif
};

std::ostream& operator << (std::ostream& os, const vector<size_t>& vec);

// CUDA: use 512 threads per block
#define CUDA_THREAD_NUM  512

// CUDA: number of blocks for threads.
#define CUDA_BLOCK_NUM(N) (((N) + CUDA_THREAD_NUM - 1) / CUDA_THREAD_NUM)

//
#define CUDNN_CHECK(x)      st(if((x) != CUDNN_STATUS_SUCCESS) { \
                                LOG(FATAL) << TO_STRING(x); \
                            })

}

#endif //! ALCHEMY_CORE_COMMON_H
