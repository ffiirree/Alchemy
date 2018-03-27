#include <algorithm>
#include <glog/logging.h>
#include "euclidean_loss_layer.h"
#include "math/math_op.h"

namespace alchemy {

template<typename T>
void EuclideanLossLayer<T>::setup(const vector<Blob<T> *> &input,
                                  const vector<Blob<T> *> &output)
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
void EuclideanLossLayer<T>::ForwardCPU(const vector<Blob<T>*>& input,
                                       const vector<Blob<T>*>& output)
{
    auto count = input[0]->count();
    //! output - label
    vector_sub(count, input[0]->data_cptr(), input[1]->data_cptr(), diff_.mutable_cptr());
    //! dot = sum_(a - y)^2
    T dot = vector_dot(count, diff_.cptr(), diff_.cptr());
    //! loss = dot/2n
    auto loss = dot / (diff_.shape(2) * (T)2);
    output[0]->mutable_data_cptr()[0] = loss;
}

template<typename T>
void EuclideanLossLayer<T>::BackwardCPU(const vector<Blob<T>*>& input,
                                        const vector<Blob<T>*>& output)
{
    auto count = input[0]->count();
    vector_copy(count, diff_.cptr(), input[0]->mutable_diff_cptr());
    vector_scal(count, (T)1.0/input[0]->shape(0), input[0]->mutable_diff_cptr());
}

template class EuclideanLossLayer<float>;
template class EuclideanLossLayer<double>;

}