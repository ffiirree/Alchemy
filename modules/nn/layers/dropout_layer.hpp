#include "math/math_op.h"

namespace alchemy {

template <typename Device, typename T>
void DropoutLayer<Device, T>::setup(const vector<container *> &input,
                            const vector<container *> &output)
{
    output[0]->reshape(input[0]->shape());

    filter_.reshape(input[0]->shape());
}

template <typename Device, typename T>
void DropoutLayer<Device, T>::ForwardCPU(const vector<container *> &input,
                                 const vector<container *> &output)
{
    auto count = input[0]->size();
    auto input_data = input[0]->data_cptr();
    auto output_data = output[0]->mutable_data_cptr();

    if(this->param_.phase() == TRAIN) {
        Filler<Device, T>::bernoulli_fill(filter_.size(), filter_.mutable_cptr(), 0.5);
        const auto filter_data = filter_.cptr();

        for(size_t i = 0; i < count; ++i) {
            output_data[i] = input_data[i] * filter_data[i];
        }
    }
    else{
        vector_copy(count, input_data, output_data);
    }
}

template <typename Device, typename T>
void DropoutLayer<Device, T>::BackwardCPU(const vector<container *> &input,
                                  const vector<container *> &output)
{
    auto count = input[0]->size();
    auto input_diff = input[0]->mutable_diff_cptr();
    auto output_diff = output[0]->diff_cptr();

    if(this->param_.phase() == TRAIN) {
        const auto filter_data = filter_.cptr();

        for(size_t i = 0; i < count; ++i) {
            input_diff[i] = output_diff[i] * filter_data[i];
        }
    }
    else {
        vector_copy(count, output_diff, input_diff);
    }
}
}