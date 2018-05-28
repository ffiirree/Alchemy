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

    void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output) override;

    void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override { }

#ifdef __CUDACC__
    void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override { }
#endif //! __CUDACC__
};
}

#include "accuracy_layer.hpp"
#ifdef __CUDACC__
#include "accuracy_layer.cuh"
#endif//! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_ACCURACY_H
