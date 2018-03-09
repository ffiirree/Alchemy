#ifndef ALCHEMY_CORE_TENSOR_H
#define ALCHEMY_CORE_TENSOR_H

#include "core/common.h"
#include "util/memory.h"

namespace alchemy {

template <typename T>
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(const vector<int>& shape);
    Tensor(const Tensor&) = default;
    Tensor&operator=(const Tensor&) = default;
    ~Tensor() = default;

    inline vector<int> shape() const { return shape_; }
    inline int shape(int axis) const { assert((unsigned)axis < this->shape_.size());  return shape_[axis]; }

    void reshape(const vector<int>& shape);

    inline int count() const { return count_; }
    inline int count(int start, int end) const {
        assert((unsigned)start < (unsigned)shape_.size());
        assert((unsigned)end <= (unsigned)shape_.size());
        assert((unsigned)start < (unsigned)end);

        int result = 1;
        for(auto i = start; i < end; ++i) {
            result *= shape_[i];
        }
        return result;
    }

    inline bool empty() const { return count_ == 0; }

    inline T * cptr() const { return reinterpret_cast<T *>(data_->cptr()); }
    inline T * gptr() const { return reinterpret_cast<T *>(data_->gptr()); }
private:
    shared_ptr<Memory> data_;
    vector<int> shape_;

    int count_ = 0;
};

template <typename T> Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b);
template <typename T> Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b);

template <typename T> Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b);
template <typename T> Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b);

}

#include "tensor.hpp"

#endif //! ALCHEMY_CORE_TENSOR_H
