#include "pooling_layer.h"

namespace alchemy {

template<typename T>
void PoolingLayer<T>::ForwardGPU(const vector<Tensor<T> *> &input,
                                 const vector<Tensor<T> *> &output)
{

}

template<typename T>
void PoolingLayer<T>::BackwardGPU(const vector<Tensor<T> *> &input,
                                  const vector<Tensor<T> *> &output)
{

}

template void PoolingLayer<float>::ForwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
template void PoolingLayer<double>::ForwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
template void PoolingLayer<float>::BackwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
template void PoolingLayer<double>::BackwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
}