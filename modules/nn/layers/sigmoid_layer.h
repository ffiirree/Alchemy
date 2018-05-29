#ifndef ALCHEMY_NN_LAYERS_SIGMOID_LAYER_H
#define ALCHEMY_NN_LAYERS_SIGMOID_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename Device, typename T>
class SigmoidLayer: public Layer<Device, T> {
public:
    using container = Blob<Device, T>;

    SigmoidLayer() = default;
    explicit SigmoidLayer(const LayerParameter& param) : Layer<Device, T>(param) { }
    virtual ~SigmoidLayer() = default;

    void setup(const vector<container *> &input, const vector<container *> &output) override;

    void Forward(const vector<container *> &input, const vector<container *> &output) override;
    void Backward(const vector<container *> &input, const vector<container *> &output) override;
};

template <typename Device, typename T>
void SigmoidLayer<Device, T>::setup(const vector<container *> &input,
                                    const vector<container *> &output)
{
    output[0]->reshape(input[0]->shape());
}

template <typename Device, typename T>
void SigmoidLayer<Device, T>::Forward(const vector<SigmoidLayer::container *> &input,
                                      const vector<SigmoidLayer::container *> &output)
{
    Sigmoid(input[0]->data(), output[0]->data());
}

template <typename Device, typename T>
void SigmoidLayer<Device, T>::Backward(const vector<SigmoidLayer::container *> &input,
                                       const vector<SigmoidLayer::container *> &output)
{
    SigmoidGrad(output[0]->data(), output[0]->diff(), input[0]->diff());
}
}
#endif //! ALCHEMY_NN_LAYERS_SIGMOID_LAYER_H
