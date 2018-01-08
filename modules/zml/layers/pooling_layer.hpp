#ifndef _ZML_POOLING_LAYER_HPP
#define _ZML_POOLING_LAYER_HPP

#include <zml/layer.hpp>

namespace z {
template <typename T>
class PoolingLayer : public Layer<T> {
    using container_type = Tensor<T>;
public:
    PoolingLayer() = default;
    explicit PoolingLayer(const LayerParameter&param): param_(param), pooling_param_(param.pooling_param()) { }
    PoolingLayer(const PoolingLayer&) = delete;
    PoolingLayer&operator=(const PoolingLayer&) = delete;
    virtual ~PoolingLayer() = default;

    inline LayerParameter parameter() const { return param_; }

    virtual void setup(const vector<container_type*>&input, const vector<container_type*>&output);

    virtual void ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);

private:
    LayerParameter param_;
    PoolingParameter pooling_param_;

    Tensor<size_t> max_idx_;
};

}


#endif //! _ZML_POOLING_LAYER_HPP
