#ifndef ALCHEMY_NN_BLOB_H
#define ALCHEMY_NN_BLOB_H

#include "core/tensor.h"

namespace alchemy {

template <typename T>
struct Blob {
public:
    Blob() = default;
    explicit Blob(const vector<int>& shape);
    Blob(const Blob&) = default;
    Blob&operator=(const Blob&) = default;
    ~Blob() = default;

    void reshape(const vector<int>& shape);

    inline auto shape() const { return data_.shape(); }
    inline auto shape(int axis) { return data_.shape(axis); }
    inline auto count() const { return data_.count(); }
    inline auto count(int start, int end) const { return data_.count(start, end); }
    inline auto num() const { return data_.shape(0); }

    inline auto data() const { return data_; }
    inline auto diff() const { return diff_; }

    inline T * data_cptr() const { return data_.cptr(); }
    inline T * data_gptr() const { return data_.gptr(); }
    inline T * diff_cptr() const { return diff_.cptr(); }
    inline T * diff_gptr() const { return diff_.gptr(); }

private:
    Tensor<T> data_;
    Tensor<T> diff_;
};

template<typename T>
Blob<T>::Blob(const vector<int> &shape)
{
    data_.reshape(shape);
    diff_.reshape(shape);
}

template<typename T>
void Blob<T>::reshape(const vector<int> &shape)
{
    data_.reshape(shape);
    diff_.reshape(shape);
}

}

#endif //! ALCHEMY_NN_BLOB_H
