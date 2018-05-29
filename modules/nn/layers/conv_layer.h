#ifndef ALCHEMY_NN_LAYERS_CONV_LAYER_H
#define ALCHEMY_NN_LAYERS_CONV_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename Device, typename T>
class ConvolutionLayer : public Layer<Device, T> {
public:
    using container = Blob<Device, T>;
    
    ConvolutionLayer() = default;
    explicit ConvolutionLayer(const LayerParameter&param)
            : Layer<Device, T>(param), conv_param_(param.conv_param()),
              kernel_(new Blob<Device, T>), bias_(new Blob<Device, T>) { }
    virtual ~ConvolutionLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void ForwardCPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardCPU(const vector<container *>& input, const vector<container *>& output) override;

#ifdef __CUDACC__
    void ForwardGPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardGPU(const vector<container *>& input, const vector<container *>& output) override;
#endif //! __CUDACC__

protected:
    ConvolutionParameter conv_param_;
    
    shared_ptr<Blob<Device, T>> kernel_;
    shared_ptr<Blob<Device, T>> bias_;
    Tensor<Device, T> biaser_;
};
}

#include "conv_layer.hpp"
#ifdef __CUDACC__
#include "conv_layer.cuh"
#endif //! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_CONV_LAYER_H
