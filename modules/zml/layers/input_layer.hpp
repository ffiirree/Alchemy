#ifndef _ZML_INPUT_LAYER_HPP
#define _ZML_INPUT_LAYER_HPP

#include <zmatrix.h>
#include <glog/logging.h>
#include <zml/util/math_op.hpp>
#include "zml/layer.hpp"
#include "zml/layer_param.hpp"

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
    virtual void BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardGPU(const vector<container_type*>& input, const vector<container_type*>& output);
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
void InputLayer<T>::setup(const vector<container_type *> &input, const vector<container_type *> &output)
{
    LOG(INFO) << "Setting up " << param_.name();

    output[0]->reshape({ static_cast<int>(input_param_.batch_size()),
                         input_param_.data()[0].first.channels(),
                         input_param_.data()[0].first.rows,
                         input_param_.data()[0].first.cols
                       });
    LOG(INFO) << "output #0: "  << output[0]->shape();
    output[1]->reshape({ static_cast<int>(input_param_.batch_size()), 1, 10, 1 }); //TODO: label 暂时这样写着
    LOG(INFO) << "output #1: " << output[1]->shape();

    vector_set(output[0]->count(), T(0), output[0]->data());

    //
    data_ = input_param_.data();
    data_num_ = data_.size();
}

template<typename T>
void InputLayer<T>::ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output)
{
    /// data
    auto data_ptr = output[0]->data();
    auto data_count = data_[0].first.size();
    auto data_size = data_count * sizeof(T);

    /// label
    auto label_ptr = output[1]->data();
    auto label_size = 10 * sizeof(T);

    for(size_t i = 0; i < input_param_.batch_size(); ++i, ++index_) {
        index_ %= data_num_;
        if(!index_) shuffle();

        auto& item = data_[index_];

        const auto& image = _Matrix<T>(item.first) * input_param_.scale();
        memmove(data_ptr, image.data, data_size);
        data_ptr += data_count;

        _Matrix<T> temp(10, 1, 1, (T)0);
        temp.at(item.second) = 1;
        memmove(label_ptr, temp.data, label_size);
        label_ptr += 10;
    }
}

template<typename T>
void InputLayer<T>::BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output)
{
}
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
