#include <algorithm>
#include <glog/logging.h>
#include <math/math_op.h>
#include "accuracy_layer.h"

namespace alchemy {

template<typename T>
void AccuracyLayer<T>::setup(const vector<Tensor<T> *> &input,
                             const vector<Tensor<T> *> &output)
{
    LOG(INFO) << "Setting up " << this->param_.name();
    LOG(INFO) << "input  #0: "  << input[0]->shape();
    LOG(INFO) << "input  #1: "  << input[1]->shape();

    output[0]->reshape({ 3 });
    LOG(INFO) << "output #0: "  << output[0]->shape();
    vector_set(output[0]->count(), (T)0., output[0]->cpu_data());
}

template<typename T>
void AccuracyLayer<T>::ForwardCPU(const vector<Tensor<T> *> &input,
                                  const vector<Tensor<T> *> &output)
{
    auto size = input[0]->shape(2) * input[0]->shape(3);
    auto o_ptr = input[0]->cpu_data();
    auto g_ptr = input[1]->cpu_data();
    int result_ = 0;
    for(auto i = 0; i < input[0]->shape(0); ++i) {
        // test
        auto o_iter = std::max_element(o_ptr + i * size, o_ptr + i * size + size);
        auto g_iter = std::max_element(g_ptr + i * size, g_ptr + i * size + size);
        if(std::distance(o_ptr + i * size, o_iter) == std::distance(g_ptr + i * size, g_iter)) {
            result_++;
        }
    }

    output[0]->cpu_data()[1] += result_;
    output[0]->cpu_data()[2] += input[0]->shape(0);
    output[0]->cpu_data()[0] = output[0]->cpu_data()[1] / output[0]->cpu_data()[2];
}

template class AccuracyLayer<float>;
template class AccuracyLayer<double>;
}
