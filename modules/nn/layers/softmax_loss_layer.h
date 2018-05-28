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

    void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output) override;

    void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;

#ifdef __CUDACC__
    void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
#endif //! __CUDACC__

private:
    SoftmaxLossParameter softmax_loss_param_{};

    shared_ptr<Layer<T>> softmax_layer_;
    vector<shared_ptr<Blob<T>>> softmax_output_;
};
}

#include "softmax_loss_layer.hpp"
#ifdef __CUDACC__
#include "softmax_loss_layer.cuh"
#endif//! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_SOFTMAX_LOSS_LAYER_H
