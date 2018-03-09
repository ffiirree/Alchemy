#ifndef ALCHEMY_NN_LAYERS_SOFTMAX_LOSS_LAYER_H
#define ALCHEMY_NN_LAYERS_SOFTMAX_LOSS_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename T>
class SoftmaxLossLayer : public Layer<T> {
public:
    SoftmaxLossLayer() = default;
    explicit SoftmaxLossLayer(const LayerParameter& param)
            : Layer<T>(param), softmax_loss_param_(param.softmax_loss_param()) { }
    virtual ~SoftmaxLossLayer() = default;

    virtual void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output);

    virtual void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    virtual void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    virtual void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
#endif //! USE_CUDA

private:
    SoftmaxLossParameter softmax_loss_param_{};

    shared_ptr<Layer<T>> softmax_layer_;
    vector<shared_ptr<Blob<T>>> softmax_output_;
};
}

#endif //! ALCHEMY_NN_LAYERS_SOFTMAX_LOSS_LAYER_H
