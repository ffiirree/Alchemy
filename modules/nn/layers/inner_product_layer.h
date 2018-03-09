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

    virtual void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output);

    virtual void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    virtual void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    virtual void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
#endif //! USE_CUDA

private:
    InnerProductParameter ip_param_{};

    shared_ptr<Blob<T>> weights_;
    shared_ptr<Blob<T>> biases_;
    Tensor<T> biasmer_;

    int M_ = 0;
    int N_ = 0;
    int K_ = 0;
};
}

#endif //! ALCHEMY_NN_LAYERS_IP_LAYER_H
