#ifndef ALCHEMY_CORE_TENSOR_HPP
#define ALCHEMY_CORE_TENSOR_HPP

#include <glog/logging.h>
#include "math/math_op.h"

namespace alchemy {

template<typename Device, typename T>
Tensor<Device, T>::Tensor(const vector<size_t> &shape)
{
    reset(shape);
}

template<typename Device, typename T>
void Tensor<Device, T>::reshape(const vector<size_t> &shape)
{
    size_t size = 1;
    size = 1;
    for(const auto& i: shape) {
        size *= i;
    }

    assert(count_ == size);

    shape_ = shape;
}

template <typename Device, typename T>
void Tensor<Device, T>::reset(const vector<size_t> &shape)
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
template<typename Device, typename T>
Tensor<Device, T> operator+(const Tensor<Device, T>& a, const Tensor<Device, T>& b)
{
    return add(a, b);
}

template<typename Device, typename T>
Tensor<Device, T> operator-(const Tensor<Device, T>& a, const Tensor<Device, T>& b)
{
    return sub(a, b);
}

template<typename Device, typename T>
Tensor<Device, T> add(const Tensor<Device, T>& a, const Tensor<Device, T>& b)
{
    assert(a.shape() == b.shape());

    Tensor<Device, T> r(a.shape());

    switch(Global::mode()) {
        case Global::CPU:
            vector_add(a.size(), a.cptr(), b.cptr(), r.mutable_cptr());
            break;

        case Global::GPU:
            vector_add_gpu(a.size(), a.gptr(), b.gptr(), r.mutable_gptr());
            break;

        default:
            LOG(FATAL) << "Unknown mode!";
            break;
    }
    return r;
}
template<typename Device, typename T>
Tensor<Device, T> sub(const Tensor<Device, T>& a, const Tensor<Device, T>& b)
{
    assert(a.shape() == b.shape());

    Tensor<Device, T> r(a.shape());

    switch(Global::mode()) {
        case Global::CPU:
            vector_sub(a.size(), a.cptr(), b.cptr(), r.cptr());
            break;

        case Global::GPU:
            vector_sub_gpu(a.size(), a.gptr(), b.gptr(), r.gptr());
            break;

        default:
            LOG(FATAL) << "Unknown mode!";
            break;
    }
    return r;
}
}

#endif //! ALCHEMY_CORE_TENSOR_HPP
