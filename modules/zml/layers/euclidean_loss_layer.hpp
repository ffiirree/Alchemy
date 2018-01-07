#ifndef _ZML_EUCLIDEAN_LOSS_LAYER_HPP
#define _ZML_EUCLIDEAN_LOSS_LAYER_HPP


#include "zml/tensor.hpp"
#include "zml/layer.hpp"

namespace z {

template <typename T>
class EuclideanLossLayer: public Layer<T> {
    using container_type = Tensor<T>;
public:
    EuclideanLossLayer()= default;
    explicit EuclideanLossLayer(const LayerParameter&parameter) : param_(parameter) { }
    EuclideanLossLayer(const EuclideanLossLayer&)= delete;
    EuclideanLossLayer&operator=(const EuclideanLossLayer&)= delete;
    virtual ~EuclideanLossLayer()= default;

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

    Tensor<T> diff_;
};

}


#endif //! _ZML_EUCLIDEAN_LOSS_LAYER_HPP
