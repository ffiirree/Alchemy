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

    virtual void setup(const vector<Tensor<T>*>&input, const vector<Tensor<T>*>&output);

    virtual void ForwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
#endif //! USE_CUDA

private:
    Tensor<T> diff_;
};
}

#endif //! ALCHEMY_NN_LAYERS_EUCLIDEAN_LOSS_LAYER_H
