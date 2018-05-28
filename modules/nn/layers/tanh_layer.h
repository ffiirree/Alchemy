#ifndef ALCHEMY_NN_LAYERS_TANH_LAYER_H
#define ALCHEMY_NN_LAYERS_TANH_LAYER_H

#include "nn/layer.h"

namespace alchemy {
template <typename T>
class TanhLayer : public Layer<T> {
public:
    TanhLayer() = default;
    explicit TanhLayer(const LayerParameter& param)
            : Layer<T>(param), tanh_param_(param.tanh_param()) { }
    virtual ~TanhLayer() = default;

    void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output) override;

    void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;

#ifdef __CUDACC__
    void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
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
