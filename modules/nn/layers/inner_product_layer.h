#ifndef ALCHEMY_NN_LAYERS_IP_LAYER_H
#define ALCHEMY_NN_LAYERS_IP_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename Device, typename T>
class InnerProductLayer: public Layer<Device, T> {
public:
    using container = Blob<Device, T>;

    InnerProductLayer() = default;
    explicit InnerProductLayer(const LayerParameter& parameter)
            : Layer<Device, T>(parameter), ip_param_(parameter.ip_param()),
              weights_(new Blob<Device, T>()), biases_(new Blob<Device, T>()) { }
    virtual ~InnerProductLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void Forward(const vector<container *>& input, const vector<container *>& output) override;
    void Backward(const vector<container *>& input, const vector<container *>& output) override;

private:
    InnerProductParameter ip_param_{};

    shared_ptr<Blob<Device, T>> weights_;
    shared_ptr<Blob<Device, T>> biases_;
    Tensor<Device, T> biasmer_;

    int M_ = 0;
    int N_ = 0;
    int K_ = 0;
};

template <typename Device, typename T>
void InnerProductLayer<Device, T>::setup(const vector<container *> &input,
                                         const vector<container *> &output)
{
    auto output_size = ip_param_.output_size();
    auto input_size = input[0]->size(1, 4);

    output[0]->reset({ input[0]->shape(0), 1, output_size, 1 });

    /// N x C x R x C
    M_ = input[0]->num();
    N_ = output_size;
    K_ = input_size;

    if(this->learnable_params_.empty()) {
        biasmer_.reset({ input[0]->shape(0) });
        weights_->reset({ output_size, input_size });
        biases_->reset({ output_size });

        this->learnable_params_.resize(2);
        this->learnable_params_[0] = std::make_tuple(weights_, ip_param_.wlr(), ip_param_.weight_decay()/input[0]->shape(0));
        this->learnable_params_[1] = std::make_tuple(biases_, ip_param_.blr(), 0.0);

        Filler<Device, T>::constant_fill(input[0]->shape(0), biasmer_.mutable_cptr(), (T)1.0);
//        vector_set(input[0]->shape(0), (T)1.0, biasmer_.mutable_cptr());

        Filler<Device, T>::fill(weights_->data(), ip_param_.weight_filler());
        Filler<Device, T>::fill(biases_->data(), ip_param_.bias_filler());
    }
}

template <typename Device, typename T>
void InnerProductLayer<Device, T>::Forward(const vector<container *>& input,
                                           const vector<container *>& output)
{

    // w * x
    MatMul(CblasNoTrans, CblasTrans,
           M_, N_, K_,
           (T)1, input[0]->data(), weights_->data(),
           (T)0, output[0]->data());

    MatMul(CblasNoTrans, CblasNoTrans,
           M_, N_, 1,
           (T)1, biasmer_, biases_->data(),
           (T)1, output[0]->data());
}


template <typename Device, typename T>
void InnerProductLayer<Device, T>::Backward(const vector<container *> &input,
                                            const vector<container *> &output)
{
    // 向前传递误差
    MatMul(CblasNoTrans, CblasNoTrans,
           M_, K_, N_,
           (T)1, output[0]->diff(), weights_->data(),
           (T)0, input[0]->diff());

    // 计算参数的更新值
    // weights
    MatMul(CblasTrans, CblasNoTrans,
           N_, K_, M_,
           (T)1, output[0]->diff(), input[0]->data(),
           (T)0, weights_->diff());

    // biases
    MVMul(CblasTrans, M_, N_,
          (T)1, output[0]->diff(), biasmer_,
          (T)0, biases_->diff());
}

} // namespace
#endif //! ALCHEMY_NN_LAYERS_IP_LAYER_H
