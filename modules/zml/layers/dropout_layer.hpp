#ifndef _ZML_DROPOUT_LAYER_HPP
#define _ZML_DROPOUT_LAYER_HPP

#include <zml/layer.hpp>

namespace z {

template <typename T>
class DropoutLayer : public Layer<T> {
    using container_type = Tensor<T>;
public:
    DropoutLayer() = default;
    explicit DropoutLayer(const LayerParameter&param)
            : param_(param), dropout_param_(param.dropout_param()) { }
    DropoutLayer(const DropoutLayer&) = delete;
    DropoutLayer&operator=(const DropoutLayer&) = delete;
    virtual ~DropoutLayer() = default;

    inline LayerParameter parameter() const { return param_; }

    virtual void setup(const vector<container_type*>&input, const vector<container_type*>&output);

    virtual void ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);

private:
    LayerParameter param_;

    DropoutParameter dropout_param_;

    Tensor<T> filter_;
};
}

#endif //! _ZML_DROPOUT_LAYER_HPP
