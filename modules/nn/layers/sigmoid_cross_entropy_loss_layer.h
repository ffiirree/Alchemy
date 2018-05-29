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

    void ForwardCPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardCPU(const vector<container *>& input, const vector<container *>& output) override;

#ifdef __CUDACC__
    void ForwardGPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardGPU(const vector<container *>& input, const vector<container *>& output) override;
#endif //! __CUDACC__

private:
    SigmoidCrossEntropyLossParameter scel_param_{};

    shared_ptr<Layer<Device, T>> sigmoid_layers_;
    vector<shared_ptr<Blob<Device, T>>> sigmoid_output_;
};
}

#include "sigmoid_cross_entropy_loss_layer.hpp"
#ifdef __CUDACC__
#include "sigmoid_cross_entropy_loss_layer.cuh"
#endif //! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_CROSS_ENTROPY_LOSS_LAYER_H
