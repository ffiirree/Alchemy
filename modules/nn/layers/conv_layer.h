#ifndef ALCHEMY_NN_LAYERS_CONV_LAYER_H
#define ALCHEMY_NN_LAYERS_CONV_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename T>
class ConvolutionLayer : public Layer<T> {
public:
    ConvolutionLayer() = default;
    explicit ConvolutionLayer(const LayerParameter&param)
            : Layer<T>(param), conv_param_(param.conv_param()),
              kernel_(new Blob<T>), bias_(new Blob<T>) { }
    virtual ~ConvolutionLayer() = default;

    void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output) override;

    void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;

#ifdef USE_CUDA
    void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
    void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) override;
#endif //! USE_CUDA

protected:
    ConvolutionParameter conv_param_;
    
    shared_ptr<Blob<T>> kernel_;
    shared_ptr<Blob<T>> bias_;
    Tensor<T> biaser_;
};
}

#include "conv_layer.hpp"
#ifdef __CUDACC__
#include "conv_layer.cuh"
#endif //! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_CONV_LAYER_H
