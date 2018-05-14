#ifndef ALCHEMY_NN_LAYERS_INPUT_LAYER_H
#define ALCHEMY_NN_LAYERS_INPUT_LAYER_H

#include <utility>
#include "nn/layer.h"

namespace alchemy {

template <typename T>
class InputLayer : public Layer<T> {
public:
    InputLayer() = default;
    explicit InputLayer(const LayerParameter& param)
            : Layer<T>(param), input_param_(param.input_param()) { }
    ~InputLayer() = default;

    virtual void setup(const vector<Blob<T>*>&input, const vector<Blob<T>*>&output);

    virtual void ForwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    virtual void BackwardCPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) { }

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output);
    virtual void BackwardGPU(const vector<Blob<T>*>& input, const vector<Blob<T>*>& output) { }
#endif

private:
    InputParameter input_param_;

    size_t index_ = 0;
    size_t data_num_{};
};
}

#endif //! ALCHEMY_NN_LAYERS_INPUT_LAYER_H
