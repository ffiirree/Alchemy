#ifndef _ZML_INPUT_LAYER_HPP
#define _ZML_INPUT_LAYER_HPP

#include "zml/layer.hpp"

namespace z {
template <typename T>
class InputLayer : public Layer<T> {
    using container_type = Tensor<T>;
public:
    InputLayer():Layer<T>() { }
    InputLayer(int num, int chs, int rows, int cols);
    InputLayer(const InputLayer&)= delete;
    InputLayer&operator=(const InputLayer&)= delete;
    ~InputLayer()= default;

    virtual void setup(const vector<container_type*>&input, const vector<container_type*>&output);

    virtual void ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);
};

template<typename T>
void InputLayer<T>::setup(const vector<container_type *> &input, const vector<container_type *> &output)
{
    LOG(INFO) << "Input Init: " << this->shape_[0] << " " << this->shape_[1] << " " << this->shape_[2] << " " << this->shape_[3];
    output[0]->reshape(this->shape_);
    output[1]->reshape({ this->shape_[0], 1, 10, 1 }); //TODO: label 暂时这样写着
}

template<typename T>
void InputLayer<T>::ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output)
{
}

template<typename T>
void InputLayer<T>::BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output)
{
}

template<typename T>
InputLayer<T>::InputLayer(int num, int chs, int rows, int cols)
    :Layer<T>()
{
    this->shape_.resize(4);
    this->shape_.at(0) = num;
    this->shape_.at(1) = chs;
    this->shape_.at(2) = rows;
    this->shape_.at(3) = cols;
}
}

#endif //! _ZML_INPUT_LAYER_HPP
