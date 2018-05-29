#include <algorithm>
#include "math/math_op.h"

namespace alchemy {
template <typename Device, typename T>
void EuclideanLossLayer<Device, T>::ForwardGPU(const vector<container *>& input,
                                       const vector<container *>& output)
{
    auto count = input[0]->size();
    //! output - label
    vector_sub_gpu(count, input[0]->data_gptr(), input[1]->data_gptr(), diff_.mutable_gptr());
    //! dot = sum_(a - y)^2
    T dot = vector_dot_gpu(count, diff_.gptr(), diff_.gptr());
    //! loss = dot/2n
    auto loss = dot / (diff_.shape(2) * (T)2);
    output[0]->mutable_data_cptr()[0] = loss;
}

template <typename Device, typename T>
void EuclideanLossLayer<Device, T>::BackwardGPU(const vector<container *>& input,
                                        const vector<container *>& output)
{
    auto count = input[0]->size();
    vector_copy_gpu(count, diff_.gptr(), input[0]->mutable_diff_gptr());
    vector_scal_gpu(count, (T)1.0/input[0]->shape(0), input[0]->mutable_diff_gptr());
}
}