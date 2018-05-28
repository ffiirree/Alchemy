#ifndef ALCHEMY_NN_LAYERS_SOFTMAX_LAYER_H
#define ALCHEMY_NN_LAYERS_SOFTMAX_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename T>
class SoftmaxLayer : public Layer<T> {
public:
    SoftmaxLayer() = default;
    explicit SoftmaxLayer(const LayerParameter& param)
            : Layer<T>(param), softmax_param_(param.softmax_param()) { }
    virtual ~SoftmaxLayer() = default;

    void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output) override;

    void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;

#ifdef __CUDACC__
    void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
#endif //! USE_CUDA

private:
    SoftmaxParameter softmax_param_;

    Blob<T> sum_;
    Blob<T> sum_multer_;
};
}

#include "softmax_layer.hpp"
#ifdef __CUDACC__
#include "softmax_layer.cuh"
#endif//! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_SOFTMAX_LAYER_H
