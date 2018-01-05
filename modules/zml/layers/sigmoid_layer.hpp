#ifndef _ZML_SIGMOID_LAYER_HPP
#define _ZML_SIGMOID_LAYER_HPP

#include "zml/layer.hpp"

namespace z {

template <typename T>
class SigmoidLayer: public Layer<T> {
    using container_type = Tensor<T>;
public:
    SigmoidLayer() = default;
    explicit SigmoidLayer(const LayerParameter& parameter) : param_(parameter) { }
    SigmoidLayer(const SigmoidLayer&)= delete;
    SigmoidLayer&operator=(const SigmoidLayer&)= delete;
    ~SigmoidLayer() = default;

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
};


}

#endif //_ZML_SIGMOID_LAYER_HPP
