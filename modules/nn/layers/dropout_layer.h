#ifndef ALCHEMY_NN_LAYERS_DROPOUT_LAYER_H
#define ALCHEMY_NN_LAYERS_DROPOUT_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename Device, typename T>
class DropoutLayer : public Layer<Device, T> {
public:
    using container = Blob<Device, T>;

    DropoutLayer() = default;
    explicit DropoutLayer(const LayerParameter&param)
            : Layer<Device, T>(param), dropout_param_(param.dropout_param()) { }
    virtual ~DropoutLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void Forward(const vector<container *>& input, const vector<container *>& output) override;
    void Backward(const vector<container *>& input, const vector<container *>& output) override;

private:
    DropoutParameter dropout_param_;
    Tensor<Device, T> filter_;
};

template <typename Device, typename T>
void DropoutLayer<Device, T>::setup(const vector<container *> &input,
                                    const vector<container *> &output)
{
    output[0]->reset(input[0]->shape());

    filter_.reset(input[0]->shape());
}

template <typename Device, typename T>
void DropoutLayer<Device, T>::Forward(const vector<container *> &input,
                                      const vector<container *> &output)
{
    if(this->param_.phase() == TRAIN) {
        Filler<Device, T>::bernoulli_fill(filter_.size(), filter_.mutable_cptr(), 0.5);
        Mul(input[0]->data(), filter_, output[0]->data());
    }
    else {
        Copy(input[0]->data(), output[0]->data());
    }
}

template <typename Device, typename T>
void DropoutLayer<Device, T>::Backward(const vector<container *> &input,
                                       const vector<container *> &output)
{

    if(this->param_.phase() == TRAIN) {
        Mul(output[0]->diff(), filter_, input[0]->diff());
    }
    else {
        Copy(output[0]->diff(), input[0]->diff());
    }
}
}
#endif //! ALCHEMY_NN_LAYERS_DROPOUT_LAYER_H
