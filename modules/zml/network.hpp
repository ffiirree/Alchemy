#ifndef _ZML_NETWORK_HPP
#define _ZML_NETWORK_HPP

#include "glog/logging.h"
#include "tensor.hpp"
#include "layer.hpp"
#include "input_layer.hpp"
#include "ip_layer.hpp"
#include "sigmoid_layer.hpp"
#include "softmax_layer.hpp"
#include "euclidean_loss_layer.hpp"


namespace z {

template <typename T>
class Network {
public:
    Network()= default;

    Network(const vector<pair<Matrix, uint8_t>>& train_data, const vector<pair<Matrix, uint8_t>>& test_data);

    Network(const vector<shared_ptr<Layer<T>>>& layers, vector<pair<Matrix, uint8_t>>& trd, const vector<pair<Matrix, uint8_t>>& td);
    ~Network();

    void sgd(const int times);
    vector<vector<pair<Matrix, uint8_t>>> split(const vector<pair<Matrix, uint8_t>> &training_data, int size) const;

    inline vector<shared_ptr<Layer<T>>> layers() const { return layers_; }
    inline vector<vector<Tensor<T>*>> inputs() const { return input_; }
    inline vector<vector<Tensor<T>*>> outputs() const { return output_; }

    void Forward();
    void Backward();

private:
    vector<shared_ptr<Layer<T>>> layers_;

    vector<shared_ptr<Tensor<T>>> tensors_;

    vector<vector<Tensor<T>*>> input_;
    vector<vector<Tensor<T>*>> output_;

    vector<pair<Matrix, uint8_t>> train_data_;
    vector<pair<Matrix, uint8_t>> test_data_;
};

template<typename T>
Network<T>::Network(const vector<pair<Matrix, uint8_t>> &train_data, const vector<pair<Matrix, uint8_t>> &test_data)
{
    train_data_ = train_data;
    test_data_ = test_data;
}

template<typename T>
Network<T>::~Network()
{
}

template<typename T>
Network<T>::Network(const vector<shared_ptr<Layer<T>>> &layers, vector<pair<Matrix, uint8_t>>& trd, const vector<pair<Matrix, uint8_t>>& td)
{
    train_data_ = trd;
    test_data_ = td;

    layers_ = layers;

    input_.resize(layers.size());
    output_.resize(layers.size());

    /// input layer
    auto layer_index = 0;
//    tensors_.resize(layers.size() + 1);

    tensors_.push_back(shared_ptr<Tensor<T>>(new Tensor<T>()));
    tensors_.push_back(shared_ptr<Tensor<T>>(new Tensor<T>()));
//    input_[layer_index].push_back(tensors_[0].get());// 输入层这个用不到
    output_[layer_index].push_back(tensors_[1].get());// data
    output_[layer_index].push_back(tensors_[0].get());// label
    layers_[layer_index].get()->setup(input_[layer_index], output_[layer_index]);
    layer_index++;

    /// hidden layers + output layer
    for(; layer_index < layers.size() - 1; ++layer_index) {
        tensors_.push_back(shared_ptr<Tensor<T>>(new Tensor<T>()));
        input_[layer_index].push_back(tensors_[layer_index].get());
        output_[layer_index].push_back(tensors_[layer_index + 1].get());

        layers_[layer_index].get()->setup(input_[layer_index], output_[layer_index]);
    }

    tensors_.push_back(shared_ptr<Tensor<T>>(new Tensor<T>()));
    /// loss layer
    input_[layer_index].push_back(tensors_[layer_index].get()); // data
    input_[layer_index].push_back(tensors_[0].get());           // label

    output_[layer_index].push_back(tensors_[layer_index + 1].get());

    layers_[layer_index].get()->setup(input_[layer_index], output_[layer_index]);
}

template<typename T>
void Network<T>::sgd(const int times)
{
    LOG(INFO) << "Train: SGD.";

    for(auto i = 0; i < times; ++i) {

        std::shuffle(train_data_.begin(), train_data_.end(), std::default_random_engine(time(nullptr)));

        auto mini_batches = split(train_data_, 10);
        for(auto& batch : mini_batches) {

            // 设置输入层的数据
//            input_[0]
            /// label
            auto label_ptr = output_[0].at(1)->data();
            auto label_size = 10 * sizeof(T);
            for(auto& item : batch) {
                _Matrix<T> temp(10, 1, 1, (T)0);
                temp.at(item.second) = 1;
                memmove(label_ptr, temp.data, label_size);
                label_ptr += 10;
            }
//            output_[0].at(0);
            /// data
            auto data_ptr = output_[0].at(0)->data();
            auto data_count = layers_[0].get()->shape(2) * layers_[0].get()->shape(3);
            auto data_size = data_count * sizeof(T);
            for(auto& item : batch) {
                auto image = _Matrix<T>(item.first)/255.0;
                memmove(data_ptr, image.data, data_size);
                data_ptr += data_count;
            }

            // 开始训练
            Forward();
            Backward();
//            std::cout << std::endl;
        }

        // Test
        // 设置数据
        auto mini_test_batches = split(test_data_, 10);
        auto hit = 0;
        for(auto& batch : mini_test_batches) {

            // 设置输入层的数据
//            input_[0]
            /// label
            auto label_ptr = output_[0].at(1)->data();
            auto label_size = 10 * sizeof(T);
            for(auto& item : batch) {
                _Matrix<T> temp(10, 1, 1, (T)0);
                temp.at(item.second) = 1;
                memmove(label_ptr, temp.data, label_size);
                label_ptr += 10;
            }
//            output_[0].at(0);
            /// data
            auto data_ptr = output_[0].at(0)->data();
            auto data_count = layers_[0].get()->shape(2) * layers_[0].get()->shape(3);
            auto data_size = data_count * sizeof(T);
            for(auto& item : batch) {
                auto image = _Matrix<T>(item.first)/255.0;
                memmove(data_ptr, image.data, data_size);
                data_ptr += data_count;
            }

            // 开始训练
            Forward();
            hit += ((EuclideanLossLayer<T> *)(layers_.back().get()))->hit();
        }
        LOG(INFO) << "Epoch {"<< i <<"} :" << hit / 10000.0;
    }
}


template<typename T>
vector<vector<pair<Matrix, uint8_t>>> Network<T>::split(const vector<pair<Matrix, uint8_t>> &training_data, int size) const
{
    vector<vector<pair<Matrix, uint8_t>>> out;
    for (size_t i = 0; i < training_data.size(); i += size) {
        out.emplace_back(training_data.begin() + i, training_data.begin() + std::min(training_data.size(), i + size));
    }
    return out;
}

template<typename T>
void Network<T>::Forward()
{
    for(auto layer_index = 0; layer_index < layers_.size(); ++layer_index) {
        layers_[layer_index]->Forward(input_[layer_index], output_[layer_index]);
    }
}

template<typename T>
void Network<T>::Backward()
{
    for(auto layer_index = layers_.size(); layer_index > 0; --layer_index) {
        layers_[layer_index - 1]->Backward(input_[layer_index - 1], output_[layer_index - 1]);
    }
}

}

#endif //! _ZML_NETWORK_HPP
