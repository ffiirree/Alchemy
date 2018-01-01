#ifndef _ZML_LAYER_HPP
#define _ZML_LAYER_HPP

#include "tensor.hpp"

namespace z {

template <typename T>
class Layer {
    using container_type = Tensor<T>;
public:
    Layer() = default;
    Layer(const Layer&)= delete;
    Layer&operator=(const Layer&)= delete;

    virtual ~Layer() = default;


    virtual void setup(const vector<container_type*>&input, const vector<container_type*>&output) = 0;

    inline vector<int> shape() const { return shape_; }
    inline int shape(const int axis) const { assert((unsigned)axis < shape_.size()); return shape_[axis]; }

    void Forward(const vector<container_type*>& input, const vector<container_type*>& output);
    void Backward(const vector<container_type*>& input, const vector<container_type*>& output);

    virtual void ForwardGPU(const vector<container_type*>& input, const vector<container_type*>& output) { ForwardCPU(input, output); }
    virtual void ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output) = 0;

    virtual void BackwardGPU(const vector<container_type*>& input, const vector<container_type*>& output) { BackwardCPU(input, output); }
    virtual void BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output) = 0;

protected:
    vector<int> shape_;

private:
    Tensor<T> loss_;
};

template<typename T>
void Layer<T>::Forward(const vector<container_type*>& input, const vector<container_type*>& output)
{
    ForwardCPU(input, output);
}

template<typename T>
void Layer<T>::Backward(const vector<container_type*>& input, const vector<container_type*>& output)
{
    BackwardCPU(input, output);
}
}

#endif //_ZML_LAYER_HPP
