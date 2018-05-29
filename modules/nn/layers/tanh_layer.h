#ifndef ALCHEMY_NN_LAYERS_TANH_LAYER_H
#define ALCHEMY_NN_LAYERS_TANH_LAYER_H

#include "nn/layer.h"

namespace alchemy {
template <typename Device, typename T>
class TanhLayer : public Layer<Device, T> {
public:
    using container = Blob<Device, T>;

    TanhLayer() = default;
    explicit TanhLayer(const LayerParameter& param)
            : Layer<Device, T>(param), tanh_param_(param.tanh_param()) { }
    virtual ~TanhLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void Forward(const vector<container *>& input, const vector<container *>& output) override;
    void Backward(const vector<container *>& input, const vector<container *>& output) override;
private:
    TanhParameter tanh_param_{};
};
template <typename Device, typename T>
void TanhLayer<Device, T>::setup(const vector<container *> &input,
                                 const vector<container *> &output)
{
    output[0]->reshape(input[0]->shape());
}

template <typename Device, typename T>
void TanhLayer<Device, T>::Forward(const vector<container *> &input,
                                   const vector<container *> &output)
{
    Tanh(input[0]->data(), output[0]->data());
}

template <typename Device, typename T>
void TanhLayer<Device, T>::Backward(const vector<container *> &input,
                                    const vector<container *> &output)
{
    TanhGrad(output[0]->data(), output[0]->diff(), input[0]->diff());
}
}
#endif //! ALCHEMY_NN_LAYERS_TANH_LAYER_H
