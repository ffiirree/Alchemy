#ifndef ALCHEMY_NN_LAYERS_EUCLIDEAN_LOSS_LAYER_H
#define ALCHEMY_NN_LAYERS_EUCLIDEAN_LOSS_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename T>
class EuclideanLossLayer: public Layer<T> {
public:
    EuclideanLossLayer() = default;
    explicit EuclideanLossLayer(const LayerParameter&parameter) : Layer<T>(parameter) { }
    virtual ~EuclideanLossLayer() = default;

    void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output) override;

    void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;

#ifdef __CUDACC__
    void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
#endif //! __CUDACC__

private:
    Tensor<T> diff_;
};
}
#include "euclidean_loss_layer.hpp"
#ifdef __CUDACC__
#include "euclidean_loss_layer.cuh"
#endif//! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_EUCLIDEAN_LOSS_LAYER_H
