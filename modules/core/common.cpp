#include "common.h"
#include <glog/logging.h>
#include <iomanip>

namespace alchemy {

Global* Global::instance_ = nullptr;
#ifdef USE_CUDA
Global::Mode Global::mode_ = Global::Mode::GPU;
#else
Global::Mode Global::mode_ = Global::Mode::CPU;
#endif

Global& Global::Instance() {
    if (instance_ == nullptr) {
        instance_ = new Global();
    }
    return *instance_;
}

Global::Global()
{
#ifdef USE_CUDA
    if(cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
        LOG(INFO) << "Create Cublas handle failed!";
    }
#endif
}

Global::~Global()
{
#ifdef USE_CUDA
    if(cublas_handle_) cublasDestroy(cublas_handle_);
#endif
}

std::ostream& operator << (std::ostream& os, const vector<int>& vec)
{
    auto size = vec.size();
    os << "[";
    for(size_t i = 0; i < size; ++i) {
        os << std::setw(3) << std::setfill(' ') << vec[i] << (i == (size - 1) ? "" : ", ");
    }
    os << "]";
    return os;
}

}