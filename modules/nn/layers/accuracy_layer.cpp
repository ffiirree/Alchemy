#include <algorithm>
#include <glog/logging.h>
#include <math/math_op.h>
#include "accuracy_layer.h"

namespace alchemy {

template<typename T>
void AccuracyLayer<T>::setup(const vector<Blob<T> *> &input,
                             const vector<Blob<T> *> &output)
{
    LOG(INFO) << "Setting up: " << this->param_.name();
    LOG(INFO) << "input  #0: "  << input[0]->shape();
    LOG(INFO) << "input  #1: "  << input[1]->shape();

    output[0]->reshape({ 3 });
    LOG(INFO) << "output #0: "  << output[0]->shape();
    vector_set(output[0]->count(), (T)0., output[0]->mutable_data_cptr());
}

template<typename T>
void AccuracyLayer<T>::ForwardCPU(const vector<Blob<T> *> &input,
                                  const vector<Blob<T> *> &output)
{
    auto size = input[0]->count(2, 4);
    auto o_ptr = input[0]->data_cptr();
    auto g_ptr = input[1]->data_cptr();
    int result_ = 0;
    for(auto i = 0; i < input[0]->shape(0); ++i) {
        // test
        auto o_iter = std::max_element(o_ptr + i * size, o_ptr + i * size + size);
        auto g_iter = std::max_element(g_ptr + i * size, g_ptr + i * size + size);
        if(std::distance(o_ptr + i * size, o_iter) == std::distance(g_ptr + i * size, g_iter)) {
            result_++;
        }
    }

    output[0]->mutable_data_cptr()[1] += result_;
    output[0]->mutable_data_cptr()[2] += input[0]->shape(0);
    output[0]->mutable_data_cptr()[0] = output[0]->data_cptr()[1] / output[0]->data_cptr()[2];
}

template class AccuracyLayer<float>;
template class AccuracyLayer<double>;
}
