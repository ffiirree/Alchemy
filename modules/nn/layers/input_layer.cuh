#ifndef ALCHEMY_NN_LAYERS_INPUT_LAYER_CUH
#define ALCHEMY_NN_LAYERS_INPUT_LAYER_CUH

#include "math/math_op.h"

namespace alchemy {

template <typename Device, typename T>
void InputLayer<Device, T>::ForwardGPU(const vector<container *> &input,
                                const vector<container *> &output)
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
}

#endif// !ALCHEMY_NN_LAYERS_INPUT_LAYER_CUH