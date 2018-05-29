#ifndef ALCHEMY_CORE_TENSOR_H
#define ALCHEMY_CORE_TENSOR_H

#include "core/common.h"
#include "util/memory.h"

namespace alchemy {

template <typename Device, typename T>
class Tensor {
public:
    using reference = T&;
    using const_reference = T const&;

    Tensor() = default;
    explicit Tensor(const vector<size_t>& shape);
    Tensor(const Tensor&) = default;
    Tensor&operator=(const Tensor&) = default;
    ~Tensor() = default;

    inline vector<size_t> shape() const { return shape_; }
    inline size_t shape(int axis) const { assert((unsigned)axis < this->shape_.size());  return shape_[axis]; }

    void reshape(const vector<size_t>& shape);

    inline size_t size() const { return count_; }
    inline size_t size(int start, int end) const {
        assert((unsigned)start < (unsigned)shape_.size());
        assert((unsigned)end <= (unsigned)shape_.size());
        assert((unsigned)start < (unsigned)end);

        size_t result = 1;
        for(auto i = start; i < end; ++i) {
            result *= shape_[i];
        }
        return result;
    }

    inline bool empty() const { return count_ == 0; }

    inline const T * cptr() const { return reinterpret_cast<const T *>(data_->cptr()); }
    inline const T * gptr() const { return reinterpret_cast<const T *>(data_->gptr()); }
    inline T * mutable_cptr() const { return reinterpret_cast<T *>(data_->mutable_cptr()); }
    inline T * mutable_gptr() const { return reinterpret_cast<T *>(data_->mutable_gptr()); }

    inline reference cat(size_t idx) { return mutable_cptr()[idx]; }
    inline const_reference cat(size_t idx) const { return cptr()[idx]; }
private:
    shared_ptr<Memory> data_;
    vector<size_t> shape_;

    size_t count_ = 0;
};

template <typename Device, typename T> Tensor<Device, T> operator+(const Tensor<Device, T>& a, const Tensor<Device, T>& b);
template <typename Device, typename T> Tensor<Device, T> operator-(const Tensor<Device, T>& a, const Tensor<Device, T>& b);

template <typename Device, typename T> Tensor<Device, T> add(const Tensor<Device, T>& a, const Tensor<Device, T>& b);
template <typename Device, typename T> Tensor<Device, T> sub(const Tensor<Device, T>& a, const Tensor<Device, T>& b);
}

#include "tensor.hpp"

#endif //! ALCHEMY_CORE_TENSOR_H
