#ifndef _ZML_COMMEN_H
#define _ZML_COMMEN_H

#include <cassert>
#include <cstddef>
#include <vector>
#include <memory>
#include <map>
#ifdef USE_CUDA
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif
#include "zcore/config.h"

using std::vector;
using std::shared_ptr;
using std::pair;
using std::string;
using std::map;
using std::tie;
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

    static Global& Instance();
#ifdef USE_CUDA
    static cublasHandle_t cublas_handle() { return Instance().cublas_handle_; }
#endif
private:
    Global();

    static Global* instance_;

#ifdef USE_CUDA
    cublasHandle_t cublas_handle_ = nullptr;
#endif
};

std::ostream& operator << (std::ostream& os, const vector<int>& vec);

}


#endif //! _ZML_COMMEN_H
