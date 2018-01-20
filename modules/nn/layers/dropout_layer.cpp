#include "dropout_layer.h"
#include <glog/logging.h>
#include "math/math_op.h"

namespace alchemy {

template<typename T>
void DropoutLayer<T>::setup(const vector<Tensor<T> *> &input,
                            const vector<Tensor<T> *> &output)
{
    LOG(INFO) << "Setting up " << this->param_.name();
    LOG(INFO) << "input  #0: "  << input[0]->shape();

    output[0]->reshape(input[0]->shape());
    LOG(INFO) << "output  #0: "  << output[0]->shape();

    filter_.reshape(input[0]->shape());
}

template<typename T>
void DropoutLayer<T>::ForwardCPU(const vector<Tensor<T> *> &input,
                                 const vector<Tensor<T> *> &output)
{
    const auto count = input[0]->count();
    const auto input_data = input[0]->cpu_data();
    auto output_data = output[0]->cpu_data();

    if(this->param_.phase() == TRAIN) {
        Filler<T>::bernoulli_fill(filter_.count(), filter_.cpu_data(), 0.5);
        const auto filter_data = filter_.cpu_data();

        for(auto i = 0; i < count; ++i) {
            output_data[i] = input_data[i] * filter_data[i];
        }
    }
    else{
        vector_copy(count, input_data, output_data);
    }
}

template<typename T>
void DropoutLayer<T>::BackwardCPU(const vector<Tensor<T> *> &input,
                                  const vector<Tensor<T> *> &output)
{
    const auto count = input[0]->count();
    auto input_diff = input[0]->cpu_diff();
    const auto output_diff = output[0]->cpu_diff();

    if(this->param_.phase() == TRAIN) {
        const auto filter_data = filter_.cpu_data();

        for(auto i = 0; i < count; ++i) {
            input_diff[i] = output_diff[i] * filter_data[i];
        }
    }
    else {
        vector_copy(count, output_diff, input_diff);
    }
}

template class DropoutLayer<float>;
template class DropoutLayer<double>;
}