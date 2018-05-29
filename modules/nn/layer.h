#ifndef ALCHEMY_NN_LAYER_H
#define ALCHEMY_NN_LAYER_H

#include <cassert>
#include <glog/logging.h>
#include "nn/blob.h"
#include "nn/layer_param.h"

namespace alchemy {

template <typename Device, typename T>
class Layer {
public:
    using container = Blob<Device, T>;
    
    Layer() = default;
    explicit Layer(const LayerParameter& param): param_(param) { }
    Layer(const Layer&) = delete;
    Layer&operator=(const Layer&) = delete;
    virtual ~Layer() = default;

    virtual void setup(const vector<container *>&input, const vector<container *>&output) = 0;

    inline LayerParameter params() const { return param_; }

    inline decltype(auto) learnable_params() const { return learnable_params_; }

    virtual void Forward(const vector<container *>& input, const vector<container *>& output);
    virtual void Backward(const vector<container *>& input, const vector<container *>& output);


    virtual void ForwardCPU(const vector<container *>& input, const vector<container *>& output) { LOG(FATAL) << ""; };
    virtual void BackwardCPU(const vector<container *>& input, const vector<container *>& output) { LOG(FATAL) << "";  };

    virtual void BackwardGPU(const vector<container *>& input, const vector<container *>& output) { BackwardCPU(input, output); }
    virtual void ForwardGPU(const vector<container *>& input, const vector<container *>& output) { ForwardCPU(input, output); }

protected:
    Tensor<Device, T> loss_;
    LayerParameter param_;

    vector<tuple<shared_ptr<Blob<Device, T>>, double, double>> learnable_params_{};
};

template <typename Device, typename T>
void Layer<Device, T>::Forward(const vector<container *>& input, const vector<container *>& output)
{
    Global::mode() == Global::CPU ? ForwardCPU(input, output) : ForwardGPU(input, output);
}

template <typename Device, typename T>
void Layer<Device, T>::Backward(const vector<container *>& input, const vector<container *>& output)
{
    Global::mode() == Global::CPU ? BackwardCPU(input, output) : BackwardGPU(input, output);
}
}

#endif //! ALCHEMY_NN_LAYER_H
