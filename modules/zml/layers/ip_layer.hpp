#ifndef _ZML_IP_LAYER_HPP
#define _ZML_IP_LAYER_HPP

extern "C" {
#include "cblas.h"
};
#include "zml/layer.hpp"
#include "zml/util/math_op.hpp"

namespace z {

template <typename T>
class InnerProductLayer: public Layer<T> {
    using container_type = Tensor<T>;
public:
    InnerProductLayer() = default;
    explicit InnerProductLayer(const LayerParameter& parameter)
            : param_(parameter), ip_param_(parameter.ip_param()) { }
    InnerProductLayer(const InnerProductLayer&)= delete;
    InnerProductLayer&operator=(const InnerProductLayer&)= delete;
    virtual ~InnerProductLayer() = default;

    inline LayerParameter parameter() const { return param_; }

    virtual void setup(const vector<container_type*>&input, const vector<container_type*>&output);

    virtual void ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardGPU(const vector<container_type*>& input, const vector<container_type*>& output);
#endif //! USE_CUDA

private:
    LayerParameter param_{};
    InnerProductParameter ip_param_{};

    shared_ptr<Tensor<T>> weights_;
    shared_ptr<Tensor<T>> biases_;
    Tensor<T> biasmer_;

    int M_ = 0;
    int N_ = 0;
    int K_ = 0;
};



}


#endif //_ZML_IP_LAYER_HPP
