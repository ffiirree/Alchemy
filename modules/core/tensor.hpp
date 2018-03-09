#ifndef ALCHEMY_CORE_TENSOR_HPP
#define ALCHEMY_CORE_TENSOR_HPP

#include <glog/logging.h>
#include "math/math_op.h"

namespace alchemy {

template<typename T>
Tensor<T>::Tensor(const vector<int> &shape)
{
    reshape(shape);
}

template<typename T>
void Tensor<T>::reshape(const vector<int> &shape)
{
    shape_ = shape;
    count_ = 1;
    for(const auto& i: shape) {
        count_ *= i;
    }
    // 分配内存
    data_.reset(new Memory(count_ * sizeof(T)));
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T> Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b)
{
    return add(a, b);
}

template <typename T> Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b)
{
    return sub(a, b);
}

template <typename T> Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b)
{
    assert(a.shape() == b.shape());

    Tensor<T> r(a.shape());

    switch(Global::mode()) {
        case Global::CPU:
            vector_add(a.count(), a.cptr(), b.cptr(), r.cptr());
            break;

        case Global::GPU:
            vector_add_gpu(a.count(), a.gptr(), b.gptr(), r.gptr());
            break;

        default:
            LOG(FATAL) << "Unknown mode!";
            break;
    }
    return r;
}
template <typename T> Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b)
{
    assert(a.shape() == b.shape());

    Tensor<T> r(a.shape());

    switch(Global::mode()) {
        case Global::CPU:
            vector_sub(a.count(), a.cptr(), b.cptr(), r.cptr());
            break;

        case Global::GPU:
            vector_sub_gpu(a.count(), a.gptr(), b.gptr(), r.gptr());
            break;

        default:
            LOG(FATAL) << "Unknown mode!";
            break;
    }
    return r;
}
}

#endif //! ALCHEMY_CORE_TENSOR_HPP
