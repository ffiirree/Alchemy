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

    void ForwardCPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardCPU(const vector<container *>& input, const vector<container *>& output) override;

#ifdef __CUDACC__
    void ForwardGPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardGPU(const vector<container *>& input, const vector<container *>& output) override;
#endif //! USE_CUDA

private:
    TanhParameter tanh_param_{};
};
}

#include "tanh_layer.hpp"
#ifdef __CUDACC__
#include "tanh_layer.cuh"
#endif//! __CUDACC__

#endif //! ALCHEMY_NN_LAYERS_TANH_LAYER_H
