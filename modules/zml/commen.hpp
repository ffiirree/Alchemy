#ifndef _ZML_COMMEN_H
#define _ZML_COMMEN_H

#include <cassert>
#include <cstddef>
#include <vector>
#include <memory>
#include <map>
#include <zcore/config.h>

#ifdef USE_CUDA
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif

using std::vector;
using std::shared_ptr;
using std::pair;
using std::string;
using std::map;
using std::tuple;

namespace z {

enum Phase {
    TRAIN, TEST, DEFAULT
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

std::ostream& operator << (std::ostream& os, const vector<int>& vec);

// CUDA: use 512 threads per block
#define CUDA_THREAD_NUM  512

// CUDA: number of blocks for threads.
#define CUDA_BLOCK_NUM(N) (((N) + CUDA_THREAD_NUM - 1) / CUDA_THREAD_NUM)

}


#endif //! _ZML_COMMEN_H
