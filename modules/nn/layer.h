#ifndef ALCHEMY_NN_LAYER_H
#define ALCHEMY_NN_LAYER_H

#include <cassert>
#include <glog/logging.h>
#include "nn/blob.h"
#include "nn/layer_param.h"

namespace alchemy {

template <typename T>
class Layer {
public:
    Layer() = default;
    explicit Layer(const LayerParameter& param): param_(param) { }
    Layer(const Layer&) = delete;
    Layer&operator=(const Layer&) = delete;
    virtual ~Layer() = default;

    virtual void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output) = 0;

    inline LayerParameter params() const { return param_; }

    inline decltype(auto) learnable_params() const { return learnable_params_; }

    void Forward(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    void Backward(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);


    virtual void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) = 0;
    virtual void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) = 0;

    virtual void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) { BackwardCPU(input, output); }
    virtual void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) { ForwardCPU(input, output); }

protected:
    Tensor<T> loss_;
    LayerParameter param_;

    vector<tuple<shared_ptr<Blob<T>>, double, double>> learnable_params_{};
};

template<typename T>
void Layer<T>::Forward(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output)
{
    Global::mode() == Global::CPU ? ForwardCPU(input, output) : ForwardGPU(input, output);
}

template<typename T>
void Layer<T>::Backward(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output)
{
    Global::mode() == Global::CPU ? BackwardCPU(input, output) : BackwardGPU(input, output);
}
}

#endif //! ALCHEMY_NN_LAYER_H
