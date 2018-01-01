#ifndef _ZML_COMMEN_H
#define _ZML_COMMEN_H

#include <cassert>
#include <cstddef>
#include <vector>
#include <memory>
#ifdef USE_CUDA
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif
#include "zcore/config.h"

using std::vector;
using std::shared_ptr;
using std::pair;

namespace z {

class Global {
public:
    Global(const Global&) = delete;
    Global&operator=(const Global&) = delete;
    ~Global();

    static Global& Instance();
#ifdef USE_CUDA
    static cublasHandle_t cublas_handle() { return Instance().cublas_handle_; }
#endif
private:
    Global();
#ifdef USE_CUDA
    cublasHandle_t cublas_handle_ = nullptr;
#endif
};

}


#endif //! _ZML_COMMEN_H
