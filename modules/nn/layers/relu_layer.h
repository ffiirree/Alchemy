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

    void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output) override;

    void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;

#ifdef __CUDACC__
    void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
#endif //! __CUDACC__

private:
    ReLuParameter relu_param_;
};
}

#include "relu_layer.hpp"
#ifdef __CUDACC__
#include "relu_layer.cuh"
#endif //! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_RELU_LAYER_H
