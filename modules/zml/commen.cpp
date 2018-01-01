#include <cublas_v2.h>
#include <glog/logging.h>
#include "commen.hpp"
#include "boost/thread.hpp"

namespace z {
// 保证线程安全
static boost::thread_specific_ptr<Global> thread_instance_;

Global& Global::Instance() {
    if (!thread_instance_.get()) {
        thread_instance_.reset(new Global());
    }
    return *(thread_instance_.get());
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

}