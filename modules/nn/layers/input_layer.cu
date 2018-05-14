#include <math/math_op.h>
#include "input_layer.h"

namespace alchemy {

template<typename T>
void InputLayer<T>::ForwardGPU(const vector<Blob<T> *> &input,
                                const vector<Blob<T> *> &output)
{
    auto batch_size = input_param_.batch_size();

    auto source = input_param_.source();
    if(!source->hasNext(static_cast<int>(batch_size))) source->reset();

    auto data_pair = source->next(static_cast<int>(batch_size));

    /// data
    cudaMemcpy(output[0]->mutable_data_gptr(),
               data_pair.first,
               batch_size * source->image_size() * sizeof(T),
               cudaMemcpyHostToDevice);

    /// label
    cudaMemcpy(output[1]->mutable_data_gptr(),
               data_pair.second,
               batch_size * source->label_size() * sizeof(T),
               cudaMemcpyHostToDevice);
}

template void InputLayer<float>::ForwardGPU(const vector<Blob<float> *> &input, const vector<Blob<float> *> &output);
template void InputLayer<double>::ForwardGPU(const vector<Blob<double> *> &input, const vector<Blob<double> *> &output);
}