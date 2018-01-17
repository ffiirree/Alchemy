#ifndef _ZML_ACCURACY_HPP
#define _ZML_ACCURACY_HPP

#include "zml/layer_param.hpp"
#include "zml/layer.hpp"

namespace z {

template <typename T>
class AccuracyLayer : public Layer<T> {
    using container_type = Tensor<T>;
public:
    AccuracyLayer() = default;
    explicit AccuracyLayer(const LayerParameter&parameter): param_(parameter) { }
    AccuracyLayer(const AccuracyLayer&) = delete;
    AccuracyLayer&operator=(const AccuracyLayer&) = delete;
    virtual ~AccuracyLayer() = default;

    inline LayerParameter parameter() const { return param_; }

    virtual void setup(const vector<container_type*>&input, const vector<container_type*>&output);

    virtual void ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output) { }

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardGPU(const vector<container_type*>& input, const vector<container_type*>& output) { }
#endif //! USE_CUDA

private:
    LayerParameter param_;
};


}

#endif //! _ZML_ACCURACY_HPP
