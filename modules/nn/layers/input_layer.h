#ifndef ALCHEMY_NN_LAYERS_INPUT_LAYER_H
#define ALCHEMY_NN_LAYERS_INPUT_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename T>
class InputLayer : public Layer<T> {
public:
    InputLayer() = default;
    explicit InputLayer(const LayerParameter& param)
            : Layer<T>(param), input_param_(param.input_param()) { }
    ~InputLayer() = default;

    virtual void setup(const vector<Tensor<T>*>&input, const vector<Tensor<T>*>&output);

    virtual void ForwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardCPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output) { }

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output);
    virtual void BackwardGPU(const vector<Tensor<T>*>& input, const vector<Tensor<T>*>& output) { }
#endif

private:
    void shuffle();
    vector<vector<pair<Matrix, uint8_t>>> split(const vector<pair<Matrix, uint8_t>> &training_data, int size) const;

    InputParameter input_param_;
    vector<pair<Matrix, uint8_t>> data_;
    vector<vector<pair<Matrix, uint8_t>>> mini_batches_;

    size_t index_ = 0;
    size_t data_num_{};
};

template<typename T>
vector<vector<pair<Matrix, uint8_t>>> InputLayer<T>::split(const vector<pair<Matrix, uint8_t>> &training_data, int size) const
{
    vector<vector<pair<Matrix, uint8_t>>> out;
    for (size_t i = 0; i < training_data.size(); i += size) {
        out.emplace_back(training_data.begin() + i, training_data.begin() + std::min(training_data.size(), i + size));
    }
    return out;
}

template<typename T>
void InputLayer<T>::shuffle()
{
    std::shuffle(data_.begin(), data_.end(), std::default_random_engine(time(nullptr)));
}

}

#endif //! ALCHEMY_NN_LAYERS_INPUT_LAYER_H
