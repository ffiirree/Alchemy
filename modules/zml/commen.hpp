#ifndef _ZML_COMMEN_H
#define _ZML_COMMEN_H

#include <cstddef>
#include <vector>
#include <memory>
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

//    static cublasHandle_t cublas_handle() { return Instance().cublas_handle_; }

private:
    Global();

//    cublasHandle_t cublas_handle_ = nullptr;
};

}


#endif //! _ZML_COMMEN_H
