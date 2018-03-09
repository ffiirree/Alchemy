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

    virtual void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output);

    virtual void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    virtual void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    virtual void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
#endif //! USE_CUDA

private:
    SoftmaxParameter softmax_param_;

    Blob<T> sum_;
    Blob<T> sum_multer_;
};
}

#endif //! ALCHEMY_NN_LAYERS_SOFTMAX_LAYER_H
