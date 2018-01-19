#ifndef ALCHEMY_NN_LAYER_H
#define ALCHEMY_NN_LAYER_H

#include <cassert>
#include "core/tensor.h"
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

    virtual void setup(const vector<Tensor<T>*>&input, const vector<Tensor<T>*>&output) = 0;

    inline LayerParameter parameter() const { return param_; }

    inline decltype(auto) learnable_params() const { return learnable_params_; }

    void Forward(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    void Backward(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);

    virtual void ForwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output) { ForwardCPU(input, output); }
    virtual void ForwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output) = 0;

    virtual void BackwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output) { BackwardCPU(input, output); }
    virtual void BackwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output) = 0;

protected:
    Tensor<T> loss_;
    LayerParameter param_;

    vector<tuple<shared_ptr<Tensor<T>>, double, double>> learnable_params_{};
};

template<typename T>
void Layer<T>::Forward(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output)
{
    Global::mode() == Global::CPU ? ForwardCPU(input, output) : ForwardGPU(input, output);
}

template<typename T>
void Layer<T>::Backward(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output)
{
    Global::mode() == Global::CPU ? BackwardCPU(input, output) : BackwardGPU(input, output);
}
}

#endif //! ALCHEMY_NN_LAYER_H
