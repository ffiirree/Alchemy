#ifndef _ZML_SOFTMAX_LAYER_HPP
#define _ZML_SOFTMAX_LAYER_HPP

#include "layer.hpp"

namespace z {

template <typename T>
class SoftmaxLayer: public Layer<T> {
    using container_type = Tensor<T>;
public:
    SoftmaxLayer() = default;
    explicit SoftmaxLayer(const Tensor<T> &input, const Tensor<T> &output) : Layer<T>(input, output) {}
    SoftmaxLayer(const SoftmaxLayer&)= delete;
    SoftmaxLayer&operator=(const SoftmaxLayer&)=delete;
    ~SoftmaxLayer() = default;

    virtual void setup(const vector<container_type*>&input, const vector<container_type*>&output);

    virtual void ForwardCPU(const Tensor<T>& input, const Tensor<T>& output);
    virtual void BackwardCPU(const Tensor<T>& input, const Tensor<T>& output);
};

template<typename T>
void SoftmaxLayer<T>::ForwardCPU(const Tensor<T> &input, const Tensor<T> &output)
{
}

template<typename T>
void SoftmaxLayer<T>::BackwardCPU(const Tensor<T> &input, const Tensor<T> &output)
{
}

template<typename T>
void SoftmaxLayer<T>::setup(const vector<container_type *> &input, const vector<container_type *> &output)
{
    output[0].get()->reshpe(this->shape_);
}

}


#endif //_ZML_SOFTMAX_LAYER_HPP
