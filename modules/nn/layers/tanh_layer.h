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

    virtual void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output);

    virtual void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    virtual void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    virtual void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
#endif //! USE_CUDA

private:
    TanhParameter tanh_param_{};
};
}

#endif //! ALCHEMY_NN_LAYERS_TANH_LAYER_H
