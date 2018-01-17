#ifndef _ZML_TANH_LAYER_HPP
#define _ZML_TANH_LAYER_HPP

#include "zml/layer.hpp"
#include "zml/layer_param.hpp"

namespace z {
template <typename T>
class TanhLayer : public Layer<T> {
    using container_type = Tensor<T>;
public:
    TanhLayer() = default;
    explicit TanhLayer(const LayerParameter& param) : param_(param), tanh_param_(param.tanh_param()) { }
    TanhLayer(const TanhLayer&)= delete;
    TanhLayer&operator=(const TanhLayer&)= delete;
    virtual ~TanhLayer() = default;

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
    TanhParameter tanh_param_{};
};
}



#endif //! _ZML_TANH_LAYER_HPP
