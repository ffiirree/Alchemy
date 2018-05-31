#ifndef ALCHEMY_NN_LAYERS_CROSS_ENTROPY_LOSS_LAYER_H
#define ALCHEMY_NN_LAYERS_CROSS_ENTROPY_LOSS_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename Device, typename T>
class SigmoidCrossEntropyLossLayer : public Layer<Device, T> {
public:
    using container = Blob<Device, T>;

    SigmoidCrossEntropyLossLayer() = default;
    explicit SigmoidCrossEntropyLossLayer(const LayerParameter& param)
            : Layer<Device, T>(param), scel_param_(param.sigmoid_cross_entropy_loss_param()) { }
    virtual ~SigmoidCrossEntropyLossLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void Forward(const vector<container *>& input, const vector<container *>& output) override;
    void Backward(const vector<container *>& input, const vector<container *>& output) override;

private:
    SigmoidCrossEntropyLossParameter scel_param_{};

    shared_ptr<Layer<Device, T>> sigmoid_layers_;
    vector<shared_ptr<Blob<Device, T>>> sigmoid_output_;
};
template <typename Device, typename T>
void SigmoidCrossEntropyLossLayer<Device, T>::setup(const vector<container *> &input,
                                                    const vector<container *> &output)
{
    LOG_IF(FATAL, input.size() < 2) << "input size: " << input.size();

    sigmoid_layers_ = shared_ptr<Layer<Device, T>>(
            new SigmoidLayer<Device, T>(
                    LayerParameter()
                            .name("<<sigmoid_cross_entropy_loss: sigmoid>>")
                            .type(SIGMOID_LAYER)
                            .sigmoid_param(
                                    SigmoidParameter()
                            )
            ));
    sigmoid_output_.push_back(shared_ptr<Blob<Device, T>>(new Blob<Device, T>()));
    sigmoid_layers_->setup(input, { sigmoid_output_[0].get() });

    output[0]->reset({ 1 });
}

template <typename Device, typename T>
void SigmoidCrossEntropyLossLayer<Device, T>::Forward(const vector<container *> &input,
                                                      const vector<container *> &output)
{
    // computes the sigmoid outputs.
    sigmoid_layers_->Forward(input, { sigmoid_output_[0].get() });

    //TODO: loss
}

template <typename Device, typename T>
void  SigmoidCrossEntropyLossLayer<Device, T>::Backward(const vector<container *> &input,
                                                        const vector<container *> &output)
{
    Sub(sigmoid_output_[0]->data(), input[1]->data(), input[0]->diff());
    Scale((T)1.0/input[0]->shape(0), input[0]->diff());
}
}
#endif //! ALCHEMY_NN_LAYERS_CROSS_ENTROPY_LOSS_LAYER_H
