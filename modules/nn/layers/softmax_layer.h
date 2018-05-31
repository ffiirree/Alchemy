#ifndef ALCHEMY_NN_LAYERS_SOFTMAX_LAYER_H
#define ALCHEMY_NN_LAYERS_SOFTMAX_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename Device, typename T>
class SoftmaxLayer : public Layer<Device, T> {
public:
    using container = Blob<Device, T>;

    SoftmaxLayer() = default;
    explicit SoftmaxLayer(const LayerParameter& param)
            : Layer<Device, T>(param), softmax_param_(param.softmax_param()) { }
    virtual ~SoftmaxLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void Forward(const vector<container *>& input, const vector<container *>& output) override;
    void Backward(const vector<container *>& input, const vector<container *>& output) override;

private:
    SoftmaxParameter softmax_param_;

    Blob<Device, T> sum_;
    Blob<Device, T> sum_multer_;
};
template <typename Device, typename T>
void SoftmaxLayer<Device, T>::setup(const vector<container *> &input,
                                    const vector<container *> &output)
{
    output[0]->reset(input[0]->shape());

    sum_.reset(input[0]->shape());
    sum_multer_.reset({ input[0]->shape(2), input[0]->shape(2) });

    Filler<Device, T>::constant_fill(sum_multer_.size(), sum_multer_.mutable_data_cptr(), (T)1.0);
}

template <typename Device, typename T>
void SoftmaxLayer<Device, T>::Forward(const vector<container *> &input,
                                      const vector<container *> &output)
{
    Copy(input[0]->data(), output[0]->data());

    //TODO: numerical issues
    // exp
    Exp(output[0]->data(), output[0]->data());
    // sum
    MatMul(CblasNoTrans, CblasNoTrans,
           input[0]->shape(0), input[0]->shape(2), input[0]->shape(2),
           (T)1., output[0]->data(), sum_multer_.data(),
           (T)0., sum_.data());
    // div
    Div(output[0]->data(), sum_.data(), output[0]->data());
}

template <typename Device, typename T>
void SoftmaxLayer<Device, T>::Backward(const vector<container *> &input,
                                       const vector<container *> &output)
{
    LOG(FATAL) << "Not implement!";
}
}
#endif //! ALCHEMY_NN_LAYERS_SOFTMAX_LAYER_H
