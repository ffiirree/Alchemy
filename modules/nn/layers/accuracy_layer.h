#ifndef ALCHEMY_NN_LAYERS_ACCURACY_H
#define ALCHEMY_NN_LAYERS_ACCURACY_H

#include "nn/layer.h"

namespace alchemy {

template <typename T>
class AccuracyLayer : public Layer<T> {
public:
    AccuracyLayer() = default;
    explicit AccuracyLayer(const LayerParameter&param) : Layer<T>(param) { }
    virtual ~AccuracyLayer() = default;

    virtual void setup(const vector<Tensor<T>*>&input, const vector<Tensor<T>*>&output);

    virtual void ForwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output) { }

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output) { }
#endif //! USE_CUDA
};
}

#endif //! ALCHEMY_NN_LAYERS_ACCURACY_H
