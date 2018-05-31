#ifndef ALCHEMY_NN_LAYERS_INPUT_LAYER_H
#define ALCHEMY_NN_LAYERS_INPUT_LAYER_H

#include <utility>
#include "nn/layer.h"

namespace alchemy {

template <typename Device, typename T>
class InputLayer : public Layer<Device, T> {
public:
    using container = Blob<Device, T>;
    
    InputLayer() = default;
    explicit InputLayer(const LayerParameter& param)
            : Layer<Device, T>(param), input_param_(param.input_param()) { }
    ~InputLayer() = default;

    void setup(const vector<container *> &input, const vector<container *> &output) override;

    void Forward(const vector<container *> &input, const vector<container *> &output) override;
    void Backward(const vector<container *> &input, const vector<container *> &output) override { }

private:
    InputParameter input_param_;
};

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

    Scale(static_cast<int>(source->size() * source->image_size()),
          (T)input_param_.scale(),
          (T*)source->images().get());
}
template <typename Device, typename T>
void InputLayer<Device, T>::Forward(const vector<container *>& input,
                                       const vector<container *>& output)
{
    auto batch_size = input_param_.batch_size();

    auto source = input_param_.source();
    if(!source->hasNext(static_cast<int>(batch_size))) source->reset();

    auto data_pair = source->next(static_cast<int>(batch_size));

    // data
    alchemy_copy(output[0]->mutable_data_ptr(),
                 data_pair.first,
                 batch_size * source->image_size() * sizeof(T));

    // label
    alchemy_copy(output[1]->mutable_data_ptr(),
                 data_pair.second,
                 batch_size * source->label_size() * sizeof(T));
}
}
#endif //! ALCHEMY_NN_LAYERS_INPUT_LAYER_H
