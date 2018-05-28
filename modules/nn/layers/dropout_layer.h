#ifndef ALCHEMY_NN_LAYERS_DROPOUT_LAYER_H
#define ALCHEMY_NN_LAYERS_DROPOUT_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename T>
class DropoutLayer : public Layer<T> {
public:
    DropoutLayer() = default;
    explicit DropoutLayer(const LayerParameter&param)
            : Layer<T>(param), dropout_param_(param.dropout_param()) { }
    virtual ~DropoutLayer() = default;

    void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output) override;

    void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;

#ifdef USE_CUDA
    void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
#endif //! USE_CUDA

private:
    DropoutParameter dropout_param_;
    Tensor<T> filter_;
};
}

#include "dropout_layer.hpp"
#ifdef __CUDACC__
#include "dropout_layer.cuh"
#endif //! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_DROPOUT_LAYER_H
