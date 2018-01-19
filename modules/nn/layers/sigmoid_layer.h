#ifndef ALCHEMY_NN_LAYERS_SIGMOID_LAYER_H
#define ALCHEMY_NN_LAYERS_SIGMOID_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename T>
class SigmoidLayer: public Layer<T> {
public:
    SigmoidLayer() = default;
    explicit SigmoidLayer(const LayerParameter& param) : Layer<T>(param) { }
    virtual ~SigmoidLayer() = default;

    virtual void setup(const vector<Tensor<T>*>&input, const vector<Tensor<T>*>&output);

    virtual void ForwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
#endif //! USE_CUDA
};
}

#endif //! ALCHEMY_NN_LAYERS_SIGMOID_LAYER_H
