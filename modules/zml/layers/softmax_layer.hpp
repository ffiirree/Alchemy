#ifndef _ZML_SOFTMAX_LAYER_HPP
#define _ZML_SOFTMAX_LAYER_HPP

#include "zml/layer.hpp"

namespace z {

template <typename T>
class SoftmaxLayer : public Layer<T> {
    using container_type = Tensor<T>;
public:
    SoftmaxLayer() = default;
    explicit SoftmaxLayer(const LayerParameter& param) : param_(param), softmax_param_(param.softmax_param()) { }
    SoftmaxLayer(const SoftmaxLayer&)= delete;
    SoftmaxLayer&operator=(const SoftmaxLayer&)=delete;
    virtual ~SoftmaxLayer() = default;

    inline LayerParameter parameter() const { return param_; }

    virtual void setup(const vector<container_type*>&input, const vector<container_type*>&output);

    virtual void ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);

private:
    LayerParameter param_;
    SoftmaxParameter softmax_param_;

    Tensor<T> sum_;
    Tensor<T> sum_multer_;
};

}


#endif //! _ZML_SOFTMAX_LAYER_HPP
