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

    virtual void setup(const vector<Tensor<T>*>&input, const vector<Tensor<T>*>&output);

    virtual void ForwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
#endif //! USE_CUDA

private:
    DropoutParameter dropout_param_;

    Tensor<T> filter_;
};
}

#endif //! ALCHEMY_NN_LAYERS_DROPOUT_LAYER_H
