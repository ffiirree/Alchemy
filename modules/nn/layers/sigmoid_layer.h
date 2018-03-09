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

    virtual void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output);

    virtual void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    virtual void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    virtual void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
#endif //! USE_CUDA
};
}

#endif //! ALCHEMY_NN_LAYERS_SIGMOID_LAYER_H
