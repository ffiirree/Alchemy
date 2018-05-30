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

    void ForwardCPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardCPU(const vector<container *>& input, const vector<container *>& output) override;

    void ForwardGPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardGPU(const vector<container *>& input, const vector<container *>& output) override;


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

        vector_set(input[0]->shape(0), (T)1.0, biasmer_.mutable_cptr());

        Filler<Device, T>::fill(weights_->data(), ip_param_.weight_filler());
        Filler<Device, T>::fill(biases_->data(), ip_param_.bias_filler());
    }
}
} // namespace

#include "inner_product_layer.hpp"
#include "inner_product_layer.cuh"
#endif //! ALCHEMY_NN_LAYERS_IP_LAYER_H
