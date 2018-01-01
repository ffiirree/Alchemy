#include <algorithm>
#include <glog/logging.h>
#include "euclidean_loss_layer.hpp"
#include "zml/util/math_op.hpp"

namespace z {

template<typename T>
void EuclideanLossLayer<T>::setup(const vector<container_type *> &input, const vector<container_type *> &output)
{
    output[0]->reshape({ 1 });
    diff_.reshape(input[0]->shape());
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

//    LOG(INFO) << "Loss: " << loss;

    hit_ = 0;
    auto size = input[0]->shape(2) * input[0]->shape(3);
    auto o_ptr = input[0]->data();
    auto g_ptr = input[1]->data();
    for(auto i = 0; i < input[0]->shape(0); ++i) {
        // test
        auto o_iter = std::max_element(o_ptr + i * size, o_ptr + i * size + size);
        auto g_iter = std::max_element(g_ptr + i * size, g_ptr + i * size + size);
        if(std::distance(o_ptr + i * size, o_iter) == std::distance(g_ptr + i * size, g_iter)) {
            hit_++;
        }
    }
}

template<typename T>
void EuclideanLossLayer<T>::BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output)
{
    auto count = input[0]->count();
    vector_copy(count, diff_.data(), input[0]->diff());
}

template<typename T>
EuclideanLossLayer<T>::EuclideanLossLayer(int num, int chs, int rows, int cols)
        :Layer<T>()
{
    this->shape_.resize(4);
    this->shape_.at(0) = num;
    this->shape_.at(1) = chs;
    this->shape_.at(2) = rows;
    this->shape_.at(3) = cols;
}

template class EuclideanLossLayer<float>;
template class EuclideanLossLayer<double>;

}