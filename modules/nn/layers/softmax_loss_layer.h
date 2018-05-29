#ifndef ALCHEMY_NN_LAYERS_SOFTMAX_LOSS_LAYER_H
#define ALCHEMY_NN_LAYERS_SOFTMAX_LOSS_LAYER_H

#include "nn/layer.h"

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

    void ForwardCPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardCPU(const vector<container *>& input, const vector<container *>& output) override;

#ifdef __CUDACC__
    void ForwardGPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardGPU(const vector<container *>& input, const vector<container *>& output) override;
#endif //! __CUDACC__

private:
    SoftmaxLossParameter softmax_loss_param_{};

    shared_ptr<Layer<Device, T>> softmax_layer_;
    vector<shared_ptr<Blob<Device, T>>> softmax_output_;
};
}

#include "softmax_loss_layer.hpp"
#ifdef __CUDACC__
#include "softmax_loss_layer.cuh"
#endif//! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_SOFTMAX_LOSS_LAYER_H
