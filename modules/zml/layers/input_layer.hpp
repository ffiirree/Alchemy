#ifndef _ZML_INPUT_LAYER_HPP
#define _ZML_INPUT_LAYER_HPP

#include <zml/layer.hpp>

namespace z {

template <typename T>
class InputLayer : public Layer<T> {
    using container_type = Tensor<T>;
public:
    InputLayer():Layer<T>() { }
    explicit InputLayer(const LayerParameter& param) : param_(param), input_param_(param.input_param()) { }
    InputLayer(const InputLayer&)= delete;
    InputLayer&operator=(const InputLayer&)= delete;
    ~InputLayer()= default;

    inline LayerParameter parameter() const { return param_; }
    vector<vector<pair<Matrix, uint8_t>>> split(const vector<pair<Matrix, uint8_t>> &training_data, int size) const;

    virtual void setup(const vector<container_type*>&input, const vector<container_type*>&output);

    void shuffle();

    virtual void ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output) { }

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardGPU(const vector<container_type*>& input, const vector<container_type*>& output) { }
#endif
private:
    LayerParameter param_;
    InputParameter input_param_;
    vector<pair<Matrix, uint8_t>> data_;
    vector<vector<pair<Matrix, uint8_t>>> mini_batches_;

    size_t index_ = 0;
    size_t data_num_;
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

#endif //! _ZML_INPUT_LAYER_HPP
