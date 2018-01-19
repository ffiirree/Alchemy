#include <algorithm>
#include <glog/logging.h>
#include "euclidean_loss_layer.h"
#include "math/math_op.h"

namespace alchemy {

template<typename T>
void EuclideanLossLayer<T>::setup(const vector<Tensor<T> *> &input,
                                  const vector<Tensor<T> *> &output)
{
    LOG(INFO) << "Setting up " << this->param_.name();
    LOG(INFO) << "input  #0: "  << input[0]->shape();
    LOG(INFO) << "input  #1: "  << input[1]->shape();

    LOG_IF(FATAL, input.size() < 2) << "input size: " << input.size();

    output[0]->reshape({ 1 });
    LOG(INFO) << "output #0: "  << output[0]->shape();

    diff_.reshape(input[0]->shape());
}

template<typename T>
void EuclideanLossLayer<T>::ForwardCPU(const vector<Tensor<T>*>& input,
                                       const vector<Tensor<T>*>& output)
{
    auto count = input[0]->count();
    //! output - label
    vector_sub(count, input[0]->cpu_data(), input[1]->cpu_data(), diff_.cpu_data());
    //! dot = sum_(a - y)^2
    T dot = vector_dot(count, diff_.cpu_data(), diff_.cpu_data());
    //! loss = dot/2n
    auto loss = dot / (diff_.shape(2) * (T)2);
    output[0]->cpu_data()[0] = loss;
}

template<typename T>
void EuclideanLossLayer<T>::BackwardCPU(const vector<Tensor<T>*>& input,
                                        const vector<Tensor<T>*>& output)
{
    auto count = input[0]->count();
    vector_copy(count, diff_.cpu_data(), input[0]->cpu_diff());
    vector_scal(count, (T)1.0/input[0]->shape(0), input[0]->cpu_diff());
}

template class EuclideanLossLayer<float>;
template class EuclideanLossLayer<double>;

}