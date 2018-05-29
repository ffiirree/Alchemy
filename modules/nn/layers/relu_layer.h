#ifndef ALCHEMY_NN_LAYERS_RELU_LAYER_H
#define ALCHEMY_NN_LAYERS_RELU_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename Device, typename T>
class ReLuLayer : public Layer<Device, T> {
public:
    using container = Blob<Device, T>;
    
    ReLuLayer() = default;
    explicit ReLuLayer(const LayerParameter& param)
            : Layer<Device, T>(param), relu_param_(param.relu_param()) { }
    ~ReLuLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void ForwardCPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardCPU(const vector<container *>& input, const vector<container *>& output) override;

#ifdef __CUDACC__
    void ForwardGPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardGPU(const vector<container *>& input, const vector<container *>& output) override;
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
