#include "dropout_layer.h"

namespace alchemy {

template<typename T>
void DropoutLayer<T>::ForwardGPU(const vector<Tensor<T> *> &input,
                                 const vector<Tensor<T> *> &output)
{
}


template<typename T>
void DropoutLayer<T>::BackwardGPU(const vector<Tensor<T> *> &input,
                                  const vector<Tensor<T> *> &output)
{
}

template void DropoutLayer<float>::ForwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
template void DropoutLayer<double>::ForwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
template void DropoutLayer<float>::BackwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
template void DropoutLayer<double>::BackwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
}