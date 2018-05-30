#include <algorithm>
#include "math/math_op.h"

namespace alchemy {

template <typename Device, typename T>
void EuclideanLossLayer<Device, T>::setup(const vector<container *> &input,
                                  const vector<container *> &output)
{
    LOG_IF(FATAL, input.size() < 2) << "input size: " << input.size();

    output[0]->reset({ 1 });
    diff_.reset(input[0]->shape());
}

template <typename Device, typename T>
void EuclideanLossLayer<Device, T>::ForwardCPU(const vector<container *>& input,
                                       const vector<container *>& output)
{
    auto count = input[0]->size();
    //! output - label
    vector_sub(count, input[0]->data_cptr(), input[1]->data_cptr(), diff_.mutable_cptr());
    //! dot = sum_(a - y)^2
    T dot = vector_dot(count, diff_.cptr(), diff_.cptr());
    //! loss = dot/2n
    auto loss = dot / (diff_.shape(2) * (T)2);
    output[0]->mutable_data_cptr()[0] = loss;
}

template<typename Device, typename T>
void EuclideanLossLayer<Device, T>::BackwardCPU(const vector<container *>& input,
                                        const vector<container *>& output)
{
    auto count = input[0]->size();
    vector_copy(count, diff_.cptr(), input[0]->mutable_diff_cptr());
    vector_scal(count, (T)1.0/input[0]->shape(0), input[0]->mutable_diff_cptr());
}
}