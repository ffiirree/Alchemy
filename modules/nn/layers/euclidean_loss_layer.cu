#include "euclidean_loss_layer.h"
#include <algorithm>
#include "math/math_op.h"

namespace alchemy {
template<typename T>
void EuclideanLossLayer<T>::ForwardGPU(const vector<Tensor<T>*>& input,
                                       const vector<Tensor<T>*>& output)
{
    auto count = input[0]->count();
    //! output - label
    vector_sub_gpu(count, input[0]->gpu_data(), input[1]->gpu_data(), diff_.gpu_data());
    //! dot = sum_(a - y)^2
    T dot = vector_dot_gpu(count, diff_.gpu_data(), diff_.gpu_data());
    //! loss = dot/2n
    auto loss = dot / (diff_.shape(2) * (T)2);
    output[0]->cpu_data()[0] = loss;
}

template<typename T>
void EuclideanLossLayer<T>::BackwardGPU(const vector<Tensor<T>*>& input,
                                        const vector<Tensor<T>*>& output)
{
    auto count = input[0]->count();
    vector_copy_gpu(count, diff_.gpu_data(), input[0]->gpu_diff());
    vector_scal_gpu(count, (T)1.0/input[0]->shape(0), input[0]->gpu_diff());
}

template void EuclideanLossLayer<float>::ForwardGPU(const vector<Tensor<float>*>& input, const vector<Tensor<float>*>& output);
template void EuclideanLossLayer<double>::ForwardGPU(const vector<Tensor<double>*>& input, const vector<Tensor<double>*>& output);
template void EuclideanLossLayer<float>::BackwardGPU(const vector<Tensor<float>*>& input, const vector<Tensor<float>*>& output);
template void EuclideanLossLayer<double>::BackwardGPU(const vector<Tensor<double>*>& input, const vector<Tensor<double>*>& output);
}