#ifndef ALCHEMY_NN_BLOB_H
#define ALCHEMY_NN_BLOB_H

#include "core/tensor.h"

namespace alchemy {

template<typename Device, typename T>
class Blob {
public:
    Blob() = default;
    explicit Blob(const vector<size_t>& shape);
    Blob(const Blob&) = default;
    Blob&operator=(const Blob&) = default;
    ~Blob() = default;

    void reshape(const vector<size_t>& shape);
    void reset(const vector<size_t>& shape);

    inline vector<size_t> shape() const { return data_.shape(); }
    inline size_t shape(int axis) { return data_.shape(axis); }
    inline size_t size() const { return data_.size(); }
    inline size_t size(int start, int end) const { return data_.size(start, end); }
    inline size_t num() const { return data_.shape(0); }

    inline Tensor<Device, T>& data() { return data_; }
    inline Tensor<Device, T> const& data() const { return data_; }
    inline Tensor<Device, T>& diff() { return diff_; }
    inline Tensor<Device, T> const& diff() const { return diff_; }

    inline const T * data_cptr() const { return data_.cptr(); }
    inline const T * data_gptr() const { return data_.gptr(); }
    inline T * mutable_data_cptr() { return data_.mutable_cptr(); }
    inline T * mutable_data_gptr() { return data_.mutable_gptr(); }

    inline const T * diff_cptr() const { return diff_.cptr(); }
    inline const T * diff_gptr() const { return diff_.gptr(); }
    inline T * mutable_diff_cptr() const { return diff_.mutable_cptr(); }
    inline T * mutable_diff_gptr() const { return diff_.mutable_gptr(); }

private:
    Tensor<Device, T> data_;
    Tensor<Device, T> diff_;
};

template<typename Device, typename T>
Blob<Device, T>::Blob(const vector<size_t> &shape)
{
    data_.reset(shape);
    diff_.reset(shape);
}
template<typename Device, typename T>
void Blob<Device, T>::reshape(const vector<size_t> &shape)
{
    data_.reshape(shape);
    diff_.reshape(shape);
}

template <typename Device, typename T>
void Blob<Device, T>::reset(const vector<size_t> &shape)
{
    data_.reset(shape);
    diff_.reset(shape);
}
}
#endif //! ALCHEMY_NN_BLOB_H
