#ifndef ALCHEMY_NN_LAYERS_EUCLIDEAN_LOSS_LAYER_H
#define ALCHEMY_NN_LAYERS_EUCLIDEAN_LOSS_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename Device, typename T>
class EuclideanLossLayer: public Layer<Device, T> {
public:
    using container = Blob<Device, T>;
    
    EuclideanLossLayer() = default;
    explicit EuclideanLossLayer(const LayerParameter&parameter) : Layer<Device, T>(parameter) { }
    virtual ~EuclideanLossLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void Forward(const vector<container *>& input, const vector<container *>& output) override;
    void Backward(const vector<container *>& input, const vector<container *>& output) override;


private:
    Tensor<Device, T> diff_;
};
template <typename Device, typename T>
void EuclideanLossLayer<Device, T>::setup(const vector<container *> &input,
                                          const vector<container *> &output)
{
    LOG_IF(FATAL, input.size() < 2) << "input size: " << input.size();

    output[0]->reset({ 1 });
    diff_.reset(input[0]->shape());
}

template <typename Device, typename T>
void EuclideanLossLayer<Device, T>::Forward(const vector<container *>& input,
                                               const vector<container *>& output)
{
    //! output - label
    Sub(input[0]->data(), input[1]->data(), diff_);
    //! dot = sum_(a - y)^2
    T dot = Dot(diff_, diff_);
    //! loss = dot/2n
    auto loss = dot / (diff_.shape(2) * (T)2);
    output[0]->mutable_data_cptr()[0] = loss;
}

template<typename Device, typename T>
void EuclideanLossLayer<Device, T>::Backward(const vector<container *>& input,
                                                const vector<container *>& output)
{
    Copy(diff_, input[0]->diff());
    Scale((T)1.0/input[0]->shape(0), input[0]->diff());
}
}
#endif //! ALCHEMY_NN_LAYERS_EUCLIDEAN_LOSS_LAYER_H
