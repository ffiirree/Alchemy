#ifndef ALCHEMY_NN_LAYERS_INPUT_LAYER_H
#define ALCHEMY_NN_LAYERS_INPUT_LAYER_H

#include <utility>

#include "nn/layer.h"

namespace alchemy {


template <typename T>
class MemoryData {
public:
    MemoryData() = default;
    MemoryData(size_t size,
               uint32_t rows, uint32_t cols, uint32_t chs, const shared_ptr<uint8_t> &images,
               uint32_t classification_num, const shared_ptr<uint8_t> &labels)
            : size_(size), rows_(rows), cols_(cols), chs_(chs),
              image_size_(rows * cols), classification_num_(classification_num)  {

        auto image_count = size_ * cols_ * rows_;
        images_.reset(new T[image_count]);

        uint8_t *src_ptr = images.get();
        T *dst_ptr = images_.get();
        for(uint32_t i = 0; i < image_count; ++i) {
            dst_ptr[i] = src_ptr[i];
        }

        labels_.reset(new T[size_ * 10]);
        src_ptr = labels.get();
        dst_ptr = labels_.get();
        for(uint32_t i = 0; i < size * 10; ++i) {
            dst_ptr[i] = src_ptr[i];
        }
    }

    inline auto size() const { return size_; }
    inline auto rows() const { return rows_; }
    inline auto cols() const { return cols_; }
    inline auto chs() const { return chs_; }
    inline auto images() const { return images_; }
    inline auto labels() const { return labels_; }
    inline auto image_size() const { return image_size_; }
    inline auto classification_num() const { return classification_num_; }
    inline auto label_size() const { return classification_num_; }

private:
    size_t size_ = 0;
    uint32_t rows_ = 0;
    uint32_t cols_ = 0;
    uint32_t chs_ = 0;
    shared_ptr<T> images_;
    shared_ptr<T> labels_;

    uint32_t image_size_ = 0;
    uint32_t classification_num_ = 0;
};

template <>
class MemoryData<uint8_t> {
public:
    MemoryData() = default;
    MemoryData(size_t size,
               uint32_t rows, uint32_t cols, uint32_t chs, shared_ptr<uint8_t> images,
               uint32_t classification_num, shared_ptr<uint8_t> labels)
            : size_(size), rows_(rows), cols_(cols), chs_(chs), images_(std::move(images)),
              labels_(std::move(labels)), image_size_(rows * cols), classification_num_(classification_num) { }

    inline auto size() const { return size_; }
    inline auto rows() const { return rows_; }
    inline auto cols() const { return cols_; }
    inline auto chs() const { return chs_; }
    inline auto images() const { return images_; }
    inline auto labels() const { return labels_; }
    inline auto image_size() const { return image_size_; }
    inline auto classification_num() const { return classification_num_; }
    inline auto label_size() const { return classification_num_; }

private:
    size_t size_ = 0;
    uint32_t rows_ = 0;
    uint32_t cols_ = 0;
    uint32_t chs_ = 0;
    shared_ptr<uint8_t> images_;
    shared_ptr<uint8_t> labels_;

    uint32_t image_size_ = 0;
    uint32_t classification_num_ = 0;
};

template <typename T>
class InputLayer : public Layer<T> {
public:
    InputLayer() = default;
    explicit InputLayer(const LayerParameter& param)
            : Layer<T>(param), input_param_(param.input_param())  {
        auto source = input_param_.source();
        data_ = {
                source.size(),
                source.rows(), source.cols(), source.chs(), source.images(),
                source.classification_num(), source.labels()
        };
    }
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
    MemoryData<T> data_;
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
//    std::shuffle(data_.begin(), data_.end(), std::default_random_engine(time(nullptr)));
}

}

#endif //! ALCHEMY_NN_LAYERS_INPUT_LAYER_H
