#include "math/math_op.h"

namespace alchemy {

template<typename T>
void DropoutLayer<T>::setup(const vector<Blob<T> *> &input,
                            const vector<Blob<T> *> &output)
{
    output[0]->reshape(input[0]->shape());

    filter_.reshape(input[0]->shape());
}

template<typename T>
void DropoutLayer<T>::ForwardCPU(const vector<Blob<T> *> &input,
                                 const vector<Blob<T> *> &output)
{
    auto count = input[0]->count();
    auto input_data = input[0]->data_cptr();
    auto output_data = output[0]->mutable_data_cptr();

    if(this->param_.phase() == TRAIN) {
        Filler<T>::bernoulli_fill(filter_.count(), filter_.mutable_cptr(), 0.5);
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
    auto count = input[0]->count();
    auto input_diff = input[0]->mutable_diff_cptr();
    auto output_diff = output[0]->diff_cptr();

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
}