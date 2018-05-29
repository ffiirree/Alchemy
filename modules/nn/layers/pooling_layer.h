#ifndef ALCHEMY_NN_LAYERS_POOLING_LAYER_H
#define ALCHEMY_NN_LAYERS_POOLING_LAYER_H

#include "nn/layer.h"

namespace alchemy {
template <typename Device, typename T>
class PoolingLayer : public Layer<Device, T> {
public:
    using container = Blob<Device, T>;
    
    PoolingLayer() = default;
    explicit PoolingLayer(const LayerParameter&param)
            : Layer<Device, T>(param), pooling_param_(param.pooling_param()) { }
    virtual ~PoolingLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void ForwardCPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardCPU(const vector<container *>& input, const vector<container *>& output) override;

#ifdef __CUDACC__
    void ForwardGPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardGPU(const vector<container *>& input, const vector<container *>& output) override;
#endif //! __CUDACC__

private:
    PoolingParameter pooling_param_;

    Tensor<Device, size_t> max_idx_;
};
}

#include "pooling_layer.hpp"
#ifdef __CUDACC__
#include "pooling_layer.cuh"
#endif //! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_POOLING_LAYER_H
