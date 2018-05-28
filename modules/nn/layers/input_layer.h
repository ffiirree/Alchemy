#ifndef ALCHEMY_NN_LAYERS_INPUT_LAYER_H
#define ALCHEMY_NN_LAYERS_INPUT_LAYER_H

#include <utility>
#include "nn/layer.h"

namespace alchemy {

template <typename T>
class InputLayer : public Layer<T> {
public:
    using container = Blob<T>;
    
    InputLayer() = default;
    explicit InputLayer(const LayerParameter& param)
            : Layer<T>(param), input_param_(param.input_param()) { }
    ~InputLayer() = default;

    void setup(const vector<container *> &input, const vector<container *> &output) override;

    void ForwardCPU(const vector<container *> &input, const vector<container *> &output) override;
    void BackwardCPU(const vector<container *> &input, const vector<container *> &output) override { }

#ifdef __CUDACC__
    void ForwardGPU(const vector<container *> &input, const vector<container *> &output) override;
    void BackwardGPU(const vector<container *> &input, const vector<container *> &output) override { }
#endif

private:
    InputParameter input_param_;

    size_t index_ = 0;
    size_t data_num_{};
};
}

#include "input_layer.hpp"
#ifdef __CUDACC__
#include "input_layer.cuh"
#endif //! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_INPUT_LAYER_H
