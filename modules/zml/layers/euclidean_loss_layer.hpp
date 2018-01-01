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
    EuclideanLossLayer(int num, int chs, int rows, int cols);
    EuclideanLossLayer(const EuclideanLossLayer&)= delete;
    EuclideanLossLayer&operator=(const EuclideanLossLayer&)= delete;
    ~EuclideanLossLayer()= default;

    inline int hit() const { return hit_; }

    virtual void setup(const vector<container_type*>&input, const vector<container_type*>&output);


    virtual void ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardGPU(const vector<container_type*>& input, const vector<container_type*>& output);
#endif //! USE_CUDA

private:
    Tensor<T> diff_;
    int hit_ = 0;
};

}


#endif //! _ZML_EUCLIDEAN_LOSS_LAYER_HPP
