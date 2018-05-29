#ifndef ALCHEMY_NN_LAYERS_DROPOUT_LAYER_H
#define ALCHEMY_NN_LAYERS_DROPOUT_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename Device, typename T>
class DropoutLayer : public Layer<Device, T> {
public:
    using container = Blob<Device, T>;
    
    DropoutLayer() = default;
    explicit DropoutLayer(const LayerParameter&param)
            : Layer<Device, T>(param), dropout_param_(param.dropout_param()) { }
    virtual ~DropoutLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void ForwardCPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardCPU(const vector<container *>& input, const vector<container *>& output) override;

#ifdef USE_CUDA
    void ForwardGPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardGPU(const vector<container *>& input, const vector<container *>& output) override;
#endif //! USE_CUDA

private:
    DropoutParameter dropout_param_;
    Tensor<Device, T> filter_;
};
}

#include "dropout_layer.hpp"
#ifdef __CUDACC__
#include "dropout_layer.cuh"
#endif //! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_DROPOUT_LAYER_H
