#ifndef _ZML_CROSS_ENTROPY_LOSS_LAYER_HPP
#define _ZML_CROSS_ENTROPY_LOSS_LAYER_HPP

#include "zml/layer.hpp"
#include "zml/layer_param.hpp"

namespace z {

template <typename T>
class SigmoidCrossEntropyLossLayer : public Layer<T> {
    using container_type = Tensor<T>;
public:
    SigmoidCrossEntropyLossLayer() = default;
    explicit SigmoidCrossEntropyLossLayer(const LayerParameter& param)
            : param_(param), scel_param_(param.sigmoid_cross_entropy_loss_param()) { }
    SigmoidCrossEntropyLossLayer(const SigmoidCrossEntropyLossLayer&) = delete;
    SigmoidCrossEntropyLossLayer&operator=(const SigmoidCrossEntropyLossLayer&) = delete;
    ~SigmoidCrossEntropyLossLayer() = default;

    inline LayerParameter parameter() const { return param_; }

    virtual void setup(const vector<container_type*>&input, const vector<container_type*>&output);

    virtual void ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);


private:
    LayerParameter param_;
    SigmoidCrossEntropyLossParameter scel_param_{};

    shared_ptr<Layer<T>> sigmoid_layers_;
    vector<shared_ptr<Tensor<T>>> sigmoid_output_;
};

}



#endif //! _ZML_CROSS_ENTROPY_LOSS_LAYER_HPP
