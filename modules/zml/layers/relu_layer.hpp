#ifndef _ZML_RELU_LAYER_HPP
#define _ZML_RELU_LAYER_HPP

#include "zml/layer.hpp"

namespace z {

template <typename T>
class ReLuLayer : public Layer<T> {
    using container_type = Tensor<T>;
public:
    ReLuLayer() = default;
    explicit ReLuLayer(const LayerParameter& param)
            : param_(param), relu_param_(param.relu_param()) { }
    ReLuLayer(const ReLuLayer&)= delete;
    ReLuLayer&operator=(const ReLuLayer&)=delete;
    ~ReLuLayer() = default;

    inline LayerParameter parameter() const { return param_; }

    virtual void setup(const vector<container_type*>&input, const vector<container_type*>&output);

    virtual void ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);

private:
    LayerParameter param_;
    ReLuParameter relu_param_;
};



}




#endif //! _ZML_RELU_LAYER_HPP
