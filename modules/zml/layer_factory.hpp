#ifndef _ZML_LAYER_FACTORY_HPP
#define _ZML_LAYER_FACTORY_HPP

#include <zml/layers/input_layer.hpp>
#include <zml/layers/accuracy_layer.hpp>
#include <zml/layers/euclidean_loss_layer.hpp>
#include <zml/layers/ip_layer.hpp>
#include <zml/layers/relu_layer.hpp>
#include <zml/layers/conv_layer.hpp>
#include <zml/layers/pooling_layer.hpp>
#include <zml/layers/sigmoid_layer.hpp>
#include <zml/layers/sigmoid_cross_entropy_loss_layer.hpp>
#include <zml/layers/tanh_layer.hpp>
#include <zml/layers/softmax_layer.hpp>
#include <zml/layers/softmax_loss_layer.hpp>

namespace z {

template <typename T>
class LayerFactory {
public:

    static shared_ptr<Layer<T>> GetLayer(const LayerParameter& param) {

        const auto type = param.type();

        switch(type) {

            case ACCURACY_LAYER:
                return shared_ptr<Layer<T>>(new AccuracyLayer<T>(param));

            case CONVOLUTION_LAYER:
                return shared_ptr<Layer<T>>(new ConvolutionLayer<T>(param));

            case EUCLIDEAN_LOSS_LAYER:
                return shared_ptr<Layer<T>>(new EuclideanLossLayer<T>(param));

            case INNER_PRODUCT_LAYER:
                return  shared_ptr<Layer<T>>(new InnerProductLayer<T>(param));

            case INPUT_LAYER:
                return shared_ptr<Layer<T>>(new InputLayer<T>(param));

            case RELU_LAYER:
                return shared_ptr<Layer<T>>(new ReLuLayer<T>(param));

            case SIGMOID_CROSS_ENTORPY_LOSS_LAYER:
                return shared_ptr<Layer<T>>(new SigmoidCrossEntropyLossLayer<T>(param));

            case SIGMOID_LAYER:
                return shared_ptr<Layer<T>>(new SigmoidLayer<T>(param));

            case SOFTMAX_LAYER:
                return shared_ptr<Layer<T>>(new SoftmaxLayer<T>(param));

            case SOFTMAX_LOSS_LAYER:
                return shared_ptr<Layer<T>>(new SoftmaxLossLayer<T>(param));

            case TANH_LAYER:
                return shared_ptr<Layer<T>>(new TanhLayer<T>(param));

            case POOLING_LAYER:
                return shared_ptr<Layer<T>>(new PoolingLayer<T>(param));

            default:
                LOG(FATAL) << "Unknown Layer type!";
                break;
        }

        return shared_ptr<Layer<T>>(nullptr);
    }

    static shared_ptr<Tensor<T>> GetSharedParam(const string& layer_name, int id = 0) {
        const auto& name = layer_name + std::to_string(id);

        auto search = shared_params_.find(name);
        if(search == shared_params_.end()) {
            shared_params_[name] = shared_ptr<Tensor<T>>(new Tensor<T>());
        }
        return shared_params_[name];
    }

private:
    static map<string, shared_ptr<Tensor<T>>> shared_params_;
};

template <typename T> map<string, shared_ptr<Tensor<T>>> LayerFactory<T>::shared_params_{};
}

#endif //! _ZML_LAYER_FACTORY_HPP
