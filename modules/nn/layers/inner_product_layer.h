#ifndef ALCHEMY_NN_LAYERS_IP_LAYER_H
#define ALCHEMY_NN_LAYERS_IP_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename T>
class InnerProductLayer: public Layer<T> {
public:
    InnerProductLayer() = default;
    explicit InnerProductLayer(const LayerParameter& parameter)
            : Layer<T>(parameter), ip_param_(parameter.ip_param()) { }
    virtual ~InnerProductLayer() = default;

    virtual void setup(const vector<Tensor<T>*>&input, const vector<Tensor<T>*>&output);

    virtual void ForwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
#endif //! USE_CUDA

private:
    InnerProductParameter ip_param_{};

    shared_ptr<Tensor<T>> weights_;
    shared_ptr<Tensor<T>> biases_;
    Tensor<T> biasmer_;

    int M_ = 0;
    int N_ = 0;
    int K_ = 0;
};
}

#endif //! ALCHEMY_NN_LAYERS_IP_LAYER_H
