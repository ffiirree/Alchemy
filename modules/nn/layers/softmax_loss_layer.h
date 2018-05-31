#ifndef ALCHEMY_NN_LAYERS_SOFTMAX_LOSS_LAYER_H
#define ALCHEMY_NN_LAYERS_SOFTMAX_LOSS_LAYER_H

#include "nn/layer.h"
#include "nn/layers/softmax_layer.h"

namespace alchemy {

template <typename Device, typename T>
class SoftmaxLossLayer : public Layer<Device, T> {
public:
    using container = Blob<Device, T>;

    SoftmaxLossLayer() = default;
    explicit SoftmaxLossLayer(const LayerParameter& param)
            : Layer<Device, T>(param), softmax_loss_param_(param.softmax_loss_param()) { }
    virtual ~SoftmaxLossLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void Forward(const vector<container *>& input, const vector<container *>& output) override;
    void Backward(const vector<container *>& input, const vector<container *>& output) override;

private:
    SoftmaxLossParameter softmax_loss_param_{};

    shared_ptr<Layer<Device, T>> softmax_layer_;
    vector<shared_ptr<Blob<Device, T>>> softmax_output_;
};

template <typename Device, typename T>
void SoftmaxLossLayer<Device, T>::setup(const vector<container *> &input,
                                        const vector<container *> &output)
{
    LOG_IF(FATAL, input.size() < 2) << "input size: " << input.size();

    softmax_layer_ = shared_ptr<Layer<Device, T>>(
            new SoftmaxLayer<Device, T>(
                    LayerParameter()
                            .name("<<softmax_loss: softmax>>")
                            .type(SOFTMAX_LAYER)
                            .softmax_param(
                                    SoftmaxParameter()
                            )
            ));

    softmax_output_.push_back(shared_ptr<Blob<Device, T>>(new Blob<Device, T>()));
    softmax_layer_->setup(input, { softmax_output_[0].get() });

    output[0]->reset({ 1 });
}


template <typename Device, typename T>
void SoftmaxLossLayer<Device, T>::Forward(const vector<container *> &input,
                                          const vector<container *> &output)
{
    softmax_layer_->Forward(input, { softmax_output_[0].get() });

    //TODO: loss
}

template <typename Device, typename T>
void SoftmaxLossLayer<Device, T>::Backward(const vector<container *> &input,
                                           const vector<container *> &output)
{
    Sub(input[0]->data(), input[1]->data(), input[0]->diff());
    Scale((T)1.0/input[0]->shape(0), input[0]->diff());
}
}
#endif //! ALCHEMY_NN_LAYERS_SOFTMAX_LOSS_LAYER_H
