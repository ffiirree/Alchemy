#include <cmath>
#include <glog/logging.h>
#include "zml/util/math_op.hpp"
#include "sigmoid_layer.hpp"

namespace z {
template <typename T>
inline T sigmoid(T value)
{
    return 1.0/(1.0 + std::exp(value));
}

template<typename T>
void SigmoidLayer<T>::ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output)
{
    auto input_data = input[0]->data();
    auto count = input[0]->count();
    auto output_data = output[0]->data();

    for(auto i = 0; i < count; ++i) {
        output_data[i] = sigmoid(-1. * input_data[i]);
    }
}

template<typename T>
void SigmoidLayer<T>::BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output)
{
    auto count = input[0]->count();
    auto output_data = output[0]->data();
    auto input_diff = input[0]->diff();
    auto output_diff = output[0]->diff();

    for(auto i = 0; i < count; ++i) {
        auto sv = output_data[i];
        input_diff[i] = output_diff[i] * sv * (1.0 - sv);
    }
}

template<typename T>
SigmoidLayer<T>::SigmoidLayer(int num, int chs, int rows, int cols)
{
    this->shape_.resize(4);
    this->shape_[0] = num;
    this->shape_[1] = chs;
    this->shape_[2] = rows;
    this->shape_[3] = cols;
}

template<typename T>
void SigmoidLayer<T>::setup(const vector<container_type *> &input, const vector<container_type *> &output)
{
    LOG(INFO) << "Sigmoid Init: " << this->shape_[0] << " " << this->shape_[1] << " " << this->shape_[2] << " " << this->shape_[3];

    output[0]->reshape(this->shape_);
}


template class SigmoidLayer<float>;
template class SigmoidLayer<double>;
}