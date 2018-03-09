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

    virtual void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output);

    virtual void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    virtual void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    virtual void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
#endif //! USE_CUDA

private:
    ReLuParameter relu_param_;
};
}

#endif //! ALCHEMY_NN_LAYERS_RELU_LAYER_H
