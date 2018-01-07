#ifndef _ZML_LAYER_FACTORY_HPP
#define _ZML_LAYER_FACTORY_HPP

#include <zml/layers/relu_layer.hpp>
#include <zml/layers/conv_layer.hpp>
#include <zml/layers/pooling_layer.hpp>
#include <zml/layers/sigmoid_cross_entropy_loss_layer.hpp>
#include <zml/layers/tanh_layer.hpp>

namespace z {

template <typename T>
class LayerFactory {
    using __LT = Layer<T>;
public:

    static shared_ptr<__LT> GetLayer(const LayerParameter& param) {

        const auto type = param.type();
        const auto name = param.name();

        auto key = make_pair(name, type);

        auto search = layers_.find(key);
        if(search == layers_.end()) {
            switch(type) {

                case ACCURACY_LAYER:
                    layers_[key] = shared_ptr<Layer<T>>(new AccuracyLayer<T>(param));
                    break;

                case CONVOLUTION_LAYER:
                    layers_[key] = shared_ptr<Layer<T>>(new ConvolutionLayer<T>(param));
                    break;

                case EUCLIDEAN_LOSS_LAYER:
                    layers_[key] = shared_ptr<Layer<T>>(new EuclideanLossLayer<T>(param));
                    break;

                case INNER_PRODUCT_LAYER:
                    layers_[key] = shared_ptr<Layer<T>>(new InnerProductLayer<T>(param));
                    break;

                case INPUT_LAYER:
                    layers_[key] = shared_ptr<Layer<T>>(new InputLayer<T>(param));
                    break;

                case RELU_LAYER:
                    layers_[key] = shared_ptr<Layer<T>>(new ReLuLayer<T>(param));
                    break;

                case SIGMOID_CROSS_ENTORPY_LOSS_LAYER:
                    layers_[key] = shared_ptr<Layer<T>>(new SigmoidCrossEntropyLossLayer<T>(param));
                    break;

                case SIGMOID_LAYER:
                    layers_[key] = shared_ptr<Layer<T>>(new SigmoidLayer<T>(param));
                    break;

                case SOFTMAX_LAYER:
                    layers_[key] = shared_ptr<Layer<T>>(new SoftmaxLayer<T>(param));
                    break;

                case TANH_LAYER:
                    layers_[key] = shared_ptr<Layer<T>>(new TanhLayer<T>(param));
                    break;

                case POOLING_LAYER:
                    layers_[key] = shared_ptr<Layer<T>>(new PoolingLayer<T>(param));
                    break;

                default:
                    LOG(INFO) << "Unknown Layer type!";
                    break;
            }
        }

        return layers_[key];
    }

private:
    static map<pair<string, LayerType>, shared_ptr<__LT>> layers_;
};

template <typename T>
map<pair<string, LayerType>, shared_ptr<Layer<T>>> LayerFactory<T>::layers_;
}

#endif //! _ZML_LAYER_FACTORY_HPP
