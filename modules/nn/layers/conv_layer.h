#ifndef ALCHEMY_NN_LAYERS_CONV_LAYER_H
#define ALCHEMY_NN_LAYERS_CONV_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename T>
class ConvolutionLayer : public Layer<T> {
public:
    ConvolutionLayer() = default;
    explicit ConvolutionLayer(const LayerParameter&param)
            : Layer<T>(param), conv_param_(param.conv_param()) { }
    virtual ~ConvolutionLayer() = default;

    virtual void setup(const vector<Tensor<T>*>&input, const vector<Tensor<T>*>&output);

    virtual void ForwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
#endif //! USE_CUDA

private:
    ConvolutionParameter conv_param_;
    
    shared_ptr<Tensor<T>> kernel_;
    shared_ptr<Tensor<T>> bias_;
    Tensor<T> biaser_;
};


}

#endif //! ALCHEMY_NN_LAYERS_CONV_LAYER_H
