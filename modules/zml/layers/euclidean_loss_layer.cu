#include <zml/util/math_op.hpp>
#include <algorithm>
#include "euclidean_loss_layer.hpp"

namespace z {

template<typename T>
void EuclideanLossLayer<T>::setup(const vector<container_type *> &input, const vector<container_type *> &output)
{
    output[0]->reshape({ 1 });
    diff_.reshape(input[0]->shape());
}

template<typename T>
void EuclideanLossLayer<T>::ForwardGPU(const vector<container_type*>& input, const vector<container_type*>& output)
{
}

template<typename T>
void EuclideanLossLayer<T>::BackwardGPU(const vector<container_type*>& input, const vector<container_type*>& output)
{
}

template<typename T>
EuclideanLossLayer<T>::EuclideanLossLayer(int num, int chs, int rows, int cols)
        :Layer<T>()
{
    this->shape_.resize(4);
    this->shape_.at(0) = num;
    this->shape_.at(1) = chs;
    this->shape_.at(2) = rows;
    this->shape_.at(3) = cols;
}

template class EuclideanLossLayer<float>;
template class EuclideanLossLayer<double>;

}