#ifndef ALCHEMY_NN_LAYERS_EUCLIDEAN_LOSS_LAYER_H
#define ALCHEMY_NN_LAYERS_EUCLIDEAN_LOSS_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename Device, typename T>
class EuclideanLossLayer: public Layer<Device, T> {
public:
    using container = Blob<Device, T>;
    
    EuclideanLossLayer() = default;
    explicit EuclideanLossLayer(const LayerParameter&parameter) : Layer<Device, T>(parameter) { }
    virtual ~EuclideanLossLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void ForwardCPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardCPU(const vector<container *>& input, const vector<container *>& output) override;

#ifdef __CUDACC__
    void ForwardGPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardGPU(const vector<container *>& input, const vector<container *>& output) override;
#endif //! __CUDACC__

private:
    Tensor<Device, T> diff_;
};
}
#include "euclidean_loss_layer.hpp"
#ifdef __CUDACC__
#include "euclidean_loss_layer.cuh"
#endif//! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_EUCLIDEAN_LOSS_LAYER_H
