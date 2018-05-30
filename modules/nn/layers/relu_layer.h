#ifndef ALCHEMY_NN_LAYERS_RELU_LAYER_H
#define ALCHEMY_NN_LAYERS_RELU_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename Device, typename T>
class ReLuLayer : public Layer<Device, T> {
public:
    using container = Blob<Device, T>;
    
    ReLuLayer() = default;
    explicit ReLuLayer(const LayerParameter& param)
            : Layer<Device, T>(param), relu_param_(param.relu_param()) { }
    ~ReLuLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void Forward(const vector<container *>& input, const vector<container *>& output) override;
    void Backward(const vector<container *>& input, const vector<container *>& output) override;
private:
    ReLuParameter relu_param_;
};

template <typename Device, typename T>
void ReLuLayer<Device, T>::setup(const vector<container *> &input,
                                 const vector<container *> &output)
{
    output[0]->reset(input[0]->shape());
}

template <typename Device, typename T>
void ReLuLayer<Device, T>::Forward(const vector<container *> &input,
                                   const vector<container *> &output)
{
    ReLU(input[0]->data(), relu_param_.alpha(), output[0]->data());
}

template <typename Device, typename T>
void ReLuLayer<Device, T>::Backward(const vector<container *> &input,
                                    const vector<container *> &output)
{
    ReLUGrad(input[0]->data(), output[0]->diff(), relu_param_.alpha(), input[0]->diff());
}
}
#endif //! ALCHEMY_NN_LAYERS_RELU_LAYER_H
