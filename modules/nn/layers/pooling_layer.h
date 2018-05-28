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

    void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output) override;

    void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;

#ifdef __CUDACC__
    void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
#endif //! __CUDACC__

private:
    PoolingParameter pooling_param_;

    Tensor<size_t> max_idx_;
};
}

#include "pooling_layer.hpp"
#ifdef __CUDACC__
#include "pooling_layer.cuh"
#endif //! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_POOLING_LAYER_H
