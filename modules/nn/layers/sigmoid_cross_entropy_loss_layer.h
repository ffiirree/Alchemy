#ifndef ALCHEMY_NN_LAYERS_CROSS_ENTROPY_LOSS_LAYER_H
#define ALCHEMY_NN_LAYERS_CROSS_ENTROPY_LOSS_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename T>
class SigmoidCrossEntropyLossLayer : public Layer<T> {
public:
    SigmoidCrossEntropyLossLayer() = default;
    explicit SigmoidCrossEntropyLossLayer(const LayerParameter& param)
            : Layer<T>(param), scel_param_(param.sigmoid_cross_entropy_loss_param()) { }
    virtual ~SigmoidCrossEntropyLossLayer() = default;

    virtual void setup(const vector<Tensor<T>*>&input, const vector<Tensor<T>*>&output);

    virtual void ForwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
#endif //! USE_CUDA

private:
    SigmoidCrossEntropyLossParameter scel_param_{};

    shared_ptr<Layer<T>> sigmoid_layers_;
    vector<shared_ptr<Tensor<T>>> sigmoid_output_;
};
}

#endif //! ALCHEMY_NN_LAYERS_CROSS_ENTROPY_LOSS_LAYER_H
