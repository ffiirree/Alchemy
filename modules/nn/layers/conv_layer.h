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

    virtual void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output);

    virtual void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    virtual void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    virtual void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
#endif //! USE_CUDA

protected:
    ConvolutionParameter conv_param_;
    
    shared_ptr<Blob<T>> kernel_;
    shared_ptr<Blob<T>> bias_;
    Tensor<T> biaser_;
};


}

#endif //! ALCHEMY_NN_LAYERS_CONV_LAYER_H
