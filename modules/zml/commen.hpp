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

    inline static void training_count(int val) { Instance().training_count_ = val; }
    inline static int training_count() { return Instance().training_count_; }

    inline static void test_count(int val) { Instance().test_count_ = val; }
    inline static int test_count() { return Instance().test_count_; }

    inline static void index(int index) { Instance().index_ = index; }
    inline static int index() { return Instance().index_; }

    static Global& Instance();
#ifdef USE_CUDA
    static cublasHandle_t cublas_handle() { return Instance().cublas_handle_; }
#endif
private:
    Global();

    static Global* instance_;

    int training_count_;
    int test_count_;
    int index_;

#ifdef USE_CUDA
    cublasHandle_t cublas_handle_ = nullptr;
#endif
};

}


#endif //! _ZML_COMMEN_H
