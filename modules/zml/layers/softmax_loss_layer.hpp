#ifndef _ZML_SOFTMAX_LOSS_LAYER_HPP
#define _ZML_SOFTMAX_LOSS_LAYER_HPP

#include <zml/layer.hpp>

namespace z {

template <typename T>
class SoftmaxLossLayer : public Layer<T> {
    using container_type = Tensor<T>;
public:
    SoftmaxLossLayer() = default;
    explicit SoftmaxLossLayer(const LayerParameter& param)
            : param_(param), softmax_loss_param_(param.softmax_loss_param()) { }
    SoftmaxLossLayer(const SoftmaxLossLayer&)= delete;
    SoftmaxLossLayer&operator=(const SoftmaxLossLayer&)=delete;
    virtual ~SoftmaxLossLayer() = default;

    inline LayerParameter parameter() const { return param_; }

    virtual void setup(const vector<container_type*>&input, const vector<container_type*>&output);

    virtual void ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);

private:
    LayerParameter param_;
    SoftmaxLossParameter softmax_loss_param_{};

    shared_ptr<Layer<T>> softmax_layer_;
    vector<shared_ptr<Tensor<T>>> softmax_output_;
};


}


#endif //! _ZML_SOFTMAX_LOSS_LAYER_HPP
