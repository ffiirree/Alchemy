#include "euclidean_loss_layer.h"
#include <algorithm>
#include "math/math_op.h"

namespace alchemy {
template<typename T>
void EuclideanLossLayer<T>::ForwardGPU(const vector<Blob<T>*>& input,
                                       const vector<Blob<T>*>& output)
{
    auto count = input[0]->count();
    //! output - label
    vector_sub_gpu(count, input[0]->data_gptr(), input[1]->data_gptr(), diff_.mutable_gptr());
    //! dot = sum_(a - y)^2
    T dot = vector_dot_gpu(count, diff_.gptr(), diff_.gptr());
    //! loss = dot/2n
    auto loss = dot / (diff_.shape(2) * (T)2);
    output[0]->mutable_data_cptr()[0] = loss;
}

template<typename T>
void EuclideanLossLayer<T>::BackwardGPU(const vector<Blob<T>*>& input,
                                        const vector<Blob<T>*>& output)
{
    auto count = input[0]->count();
    vector_copy_gpu(count, diff_.gptr(), input[0]->mutable_diff_gptr());
    vector_scal_gpu(count, (T)1.0/input[0]->shape(0), input[0]->mutable_diff_gptr());
}

template void EuclideanLossLayer<float>::ForwardGPU(const vector<Blob<float>*>& input, const vector<Blob<float>*>& output);
template void EuclideanLossLayer<double>::ForwardGPU(const vector<Blob<double>*>& input, const vector<Blob<double>*>& output);
template void EuclideanLossLayer<float>::BackwardGPU(const vector<Blob<float>*>& input, const vector<Blob<float>*>& output);
template void EuclideanLossLayer<double>::BackwardGPU(const vector<Blob<double>*>& input, const vector<Blob<double>*>& output);
}