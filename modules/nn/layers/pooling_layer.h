#ifndef ALCHEMY_NN_LAYERS_POOLING_LAYER_H
#define ALCHEMY_NN_LAYERS_POOLING_LAYER_H

#include "nn/layer.h"

namespace alchemy {
template <typename T>
class PoolingLayer : public Layer<T> {
public:
    PoolingLayer() = default;
    explicit PoolingLayer(const LayerParameter&param)
            : Layer<T>(param), pooling_param_(param.pooling_param()) { }
    virtual ~PoolingLayer() = default;

    virtual void setup(const vector<Tensor<T>*>&input, const vector<Tensor<T>*>&output);

    virtual void ForwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
#endif //! USE_CUDA

private:
    PoolingParameter pooling_param_;

    Tensor<size_t> max_idx_;
};
}

#endif //! ALCHEMY_NN_LAYERS_POOLING_LAYER_H
