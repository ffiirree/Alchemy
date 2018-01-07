#include <algorithm>
#include <glog/logging.h>
#include <iostream>
#include "euclidean_loss_layer.hpp"
#include "zml/util/math_op.hpp"

namespace z {

template<typename T>
void EuclideanLossLayer<T>::setup(const vector<container_type *> &input, const vector<container_type *> &output)
{
    output[0]->reshape({ 1 });
    diff_.reshape(input[0]->shape());

    LOG(INFO) << "Euclidean Loss Layer: { out: " << output[0]->shape() << " }";
}

template<typename T>
void EuclideanLossLayer<T>::ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output)
{
    auto count = input[0]->count();
    //! output - label
    vector_sub(count, input[0]->data(), input[1]->data(), diff_.data());
    //! dot = sum_(a - y)^2
    T dot = vector_dot(count, diff_.data(), diff_.data());
    //! loss = dot/2n
    auto loss = dot / (diff_.shape(2) * (T)2);
    output[0]->data()[0] = loss;


}

template<typename T>
void EuclideanLossLayer<T>::BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output)
{
    auto count = input[0]->count();
    vector_copy(count, diff_.data(), input[0]->diff());
}

template class EuclideanLossLayer<float>;
template class EuclideanLossLayer<double>;

}