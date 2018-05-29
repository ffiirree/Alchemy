#ifndef ALCHEMY_NN_LAYERS_ACCURACY_H
#define ALCHEMY_NN_LAYERS_ACCURACY_H

#include "nn/layer.h"

namespace alchemy {

template <typename Device, typename T>
class AccuracyLayer : public Layer<Device, T> {
public:
    using container = Blob<Device, T>;
    
    AccuracyLayer() = default;
    explicit AccuracyLayer(const LayerParameter&param) : Layer<Device, T>(param) { }
    virtual ~AccuracyLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void ForwardCPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardCPU(const vector<container *>& input, const vector<container *>& output) override { }

#ifdef __CUDACC__
    void ForwardGPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardGPU(const vector<container *>& input, const vector<container *>& output) override { }
#endif //! __CUDACC__
};
}

#include "accuracy_layer.hpp"
#ifdef __CUDACC__
#include "accuracy_layer.cuh"
#endif//! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_ACCURACY_H
