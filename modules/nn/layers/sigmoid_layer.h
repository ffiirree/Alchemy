#ifndef ALCHEMY_NN_LAYERS_SIGMOID_LAYER_H
#define ALCHEMY_NN_LAYERS_SIGMOID_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename T>
class SigmoidLayer: public Layer<T> {
public:
    using container = Blob<T>;
    
    SigmoidLayer() = default;
    explicit SigmoidLayer(const LayerParameter& param) : Layer<T>(param) { }
    virtual ~SigmoidLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void ForwardCPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardCPU(const vector<container *>& input, const vector<container *>& output) override;

#ifdef __CUDACC__
    void ForwardGPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardGPU(const vector<container *>& input, const vector<container *>& output) override;
#endif //! USE_CUDA
};
}

#include "sigmoid_layer.hpp"
#ifdef __CUDACC__
#include "sigmoid_layer.cuh"
#endif

#endif //! ALCHEMY_NN_LAYERS_SIGMOID_LAYER_H
