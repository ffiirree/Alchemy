#ifndef ALCHEMY_NN_LAYER_FACTORY_H
#define ALCHEMY_NN_LAYER_FACTORY_H

#include "nn/layers/input_layer.h"
#include "nn/layers/accuracy_layer.h"
#include "nn/layers/euclidean_loss_layer.h"
#include "nn/layers/inner_product_layer.h"
#include "nn/layers/relu_layer.h"
#include "nn/layers/conv_layer.h"
#include "nn/layers/cudnn_conv_layer.h"
#include "nn/layers/pooling_layer.h"
#include "nn/layers/sigmoid_layer.h"
#include "nn/layers/sigmoid_cross_entropy_loss_layer.h"
#include "nn/layers/tanh_layer.h"
#include "nn/layers/softmax_layer.h"
#include "nn/layers/softmax_loss_layer.h"
#include "nn/layers/dropout_layer.h"

#define REGISTER_LAYER(TYPE, NAME, CLASS, PARAMS)\
case (TYPE):\
    layers_[(NAME)] = shared_ptr<Layer<T>>(new CLASS<T>(PARAMS));\
    break

namespace alchemy {

template <typename T>
class LayerFactory {
public:
    static shared_ptr<Layer<T>> GetLayer(const LayerParameter& param) {
        auto key = param.id();
        auto type = param.type();
        auto search = layers_.find(key);

        if(search == layers_.end()) {
            switch(type) {
                REGISTER_LAYER(ACCURACY_LAYER, key, AccuracyLayer, param);
                REGISTER_LAYER(CONVOLUTION_LAYER, key, ConvolutionLayer, param);
                REGISTER_LAYER(CUDNN_CONV_LAYER, key, CuDNNConvolutionLayer, param);
                REGISTER_LAYER(DROPOUT_LAYER, key, DropoutLayer, param);
                REGISTER_LAYER(EUCLIDEAN_LOSS_LAYER, key, EuclideanLossLayer, param);
                REGISTER_LAYER(INNER_PRODUCT_LAYER, key, InnerProductLayer, param);
                REGISTER_LAYER(INPUT_LAYER, key, InputLayer, param);
                REGISTER_LAYER(RELU_LAYER, key, ReLuLayer, param);
                REGISTER_LAYER(SIGMOID_CROSS_ENTORPY_LOSS_LAYER, key, SigmoidCrossEntropyLossLayer, param);
                REGISTER_LAYER(SIGMOID_LAYER, key, SigmoidLayer, param);
                REGISTER_LAYER(SOFTMAX_LAYER, key, SoftmaxLossLayer, param);
                REGISTER_LAYER(SOFTMAX_LOSS_LAYER, key, SoftmaxLayer, param);
                REGISTER_LAYER(TANH_LAYER, key, TanhLayer, param);
                REGISTER_LAYER(POOLING_LAYER, key, PoolingLayer, param);

                default: LOG(FATAL) << "Unknown Layer type!"; break;
            }
        }

        return layers_[key];
    }

private:
    static map<string, shared_ptr<Layer<T>>> layers_;
};//! class LayerFactory

template <typename T> map<string, shared_ptr<Layer<T>>> LayerFactory<T>::layers_{};
}//! namespace
#undef REGISTER_LAYER
#endif //! ALCHEMY_NN_LAYER_FACTORY_H
