#ifndef ALCHEMY_NN_LAYERS_SOFTMAX_LAYER_H
#define ALCHEMY_NN_LAYERS_SOFTMAX_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename Device, typename T>
class SoftmaxLayer : public Layer<Device, T> {
public:
    using container = Blob<Device, T>;
    
    SoftmaxLayer() = default;
    explicit SoftmaxLayer(const LayerParameter& param)
            : Layer<Device, T>(param), softmax_param_(param.softmax_param()) { }
    virtual ~SoftmaxLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void ForwardCPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardCPU(const vector<container *>& input, const vector<container *>& output) override;

    void ForwardGPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardGPU(const vector<container *>& input, const vector<container *>& output) override;

private:
    SoftmaxParameter softmax_param_;

    Blob<Device, T> sum_;
    Blob<Device, T> sum_multer_;
};
}

#include "softmax_layer.hpp"
#include "softmax_layer.cuh"
#endif //! ALCHEMY_NN_LAYERS_SOFTMAX_LAYER_H
