#ifndef ALCHEMY_NN_LAYERS_INPUT_LAYER_HPP
#define ALCHEMY_NN_LAYERS_INPUT_LAYER_HPP

#include "math/math_op.h"

namespace alchemy {

template <typename Device, typename T>
void InputLayer<Device, T>::setup(const vector<container *> &input,
                          const vector<container *> &output)
{
    auto source = input_param_.source();

    output[0]->reset({ input_param_.batch_size(),
                         source->chs(),
                         source->rows(),
                         source->cols()
                       });
    output[1]->reset({ input_param_.batch_size(),
                         1,
                         source->classification_num(),
                         1
                       });

    data_num_ = source->size();
    vector_scal(static_cast<int>(source->size() * source->image_size()),
                (T)input_param_.scale(),
                (T*)source->images().get());
}

template <typename Device, typename T>
void InputLayer<Device, T>::ForwardCPU(const vector<container *>& input,
                               const vector<container *>& output)
{
    auto batch_size = input_param_.batch_size();

    auto source = input_param_.source();
    if(!source->hasNext(static_cast<int>(batch_size))) source->reset();

    auto data_pair = source->next(static_cast<int>(batch_size));

    /// data
    memmove(output[0]->mutable_data_cptr(),
            data_pair.first,
            batch_size * source->image_size() * sizeof(T));

//    print_cpu(output[0]->count(), output[0]->data_cptr());

    /// label
    memmove(output[1]->mutable_data_cptr(),
            data_pair.second,
            batch_size * source->label_size() * sizeof(T));

//    print_cpu(output[1]->count(), output[1]->data_cptr());
}
}

#endif// !ALCHEMY_NN_LAYERS_INPUT_LAYER_HPP