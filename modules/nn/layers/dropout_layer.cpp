#include "dropout_layer.h"
#include <glog/logging.h>
#include "math/math_op.h"

namespace alchemy {

template<typename T>
void DropoutLayer<T>::setup(const vector<Blob<T> *> &input,
                            const vector<Blob<T> *> &output)
{
    LOG(INFO) << "Setting up " << this->param_.name();
    LOG(INFO) << "input  #0: "  << input[0]->shape();

    output[0]->reshape(input[0]->shape());
    LOG(INFO) << "output  #0: "  << output[0]->shape();

    filter_.reshape(input[0]->shape());
}

template<typename T>
void DropoutLayer<T>::ForwardCPU(const vector<Blob<T> *> &input,
                                 const vector<Blob<T> *> &output)
{
    const auto count = input[0]->count();
    const auto input_data = input[0]->data_cptr();
    auto output_data = output[0]->data_cptr();

    if(this->param_.phase() == TRAIN) {
        Filler<T>::bernoulli_fill(filter_.count(), filter_.cptr(), 0.5);
        const auto filter_data = filter_.cptr();

        for(auto i = 0; i < count; ++i) {
            output_data[i] = input_data[i] * filter_data[i];
        }
    }
    else{
        vector_copy(count, input_data, output_data);
    }
}

template<typename T>
void DropoutLayer<T>::BackwardCPU(const vector<Blob<T> *> &input,
                                  const vector<Blob<T> *> &output)
{
    const auto count = input[0]->count();
    auto input_diff = input[0]->diff_cptr();
    const auto output_diff = output[0]->diff_cptr();

    if(this->param_.phase() == TRAIN) {
        const auto filter_data = filter_.cptr();

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