#ifndef ALCHEMY_NN_LAYERS_RELU_LAYER_H
#define ALCHEMY_NN_LAYERS_RELU_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename T>
class ReLuLayer : public Layer<T> {
public:
    ReLuLayer() = default;
    explicit ReLuLayer(const LayerParameter& param)
            : Layer<T>(param), relu_param_(param.relu_param()) { }
    ~ReLuLayer() = default;

    virtual void setup(const vector<Tensor<T>*>&input, const vector<Tensor<T>*>&output);

    virtual void ForwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
#endif //! USE_CUDA

private:
    ReLuParameter relu_param_;
};
}

#endif //! ALCHEMY_NN_LAYERS_RELU_LAYER_H
