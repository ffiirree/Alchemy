#ifndef _ZML_TENSOR_HPP
#define _ZML_TENSOR_HPP

#include "commen.hpp"

namespace z {

template <typename T>
class Tensor {
public:
    Tensor() = default;
    explicit Tensor(const vector<int>& shape);
    Tensor(const Tensor&) = default;
    Tensor&operator=(const Tensor&) = default;
    ~Tensor() = default;

    operator bool() { return count_ == 0; }

    void reshape(const vector<int>& shape);

    inline bool empty() const { return count_ == 0; }
    inline int count() const { return count_; }
    inline vector<int> shape() const { return shape_; }
    inline int shape(int axis) const { assert((unsigned)axis < this->shape_.size());  return shape_[axis]; }

    inline T * data() const { return reinterpret_cast<T *>(data_.get()); }
    inline T * diff() const { return reinterpret_cast<T *>(diff_.get()); }

private:
    shared_ptr<uint8_t> data_;
    shared_ptr<uint8_t> diff_;
    vector<int> shape_;

    int count_ = 0;
};

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
    data_.reset((uint8_t *)malloc(static_cast<size_t>(count_) * sizeof(T)));
    diff_.reset((uint8_t *)malloc(static_cast<size_t>(count_) * sizeof(T)));
}

}
#endif //! _ZML_TENSOR_HPP
