#include "conv_layer.h"

namespace alchemy {

template<typename T>
void ConvolutionLayer<T>::ForwardGPU(const vector<Tensor<T> *> &input,
                                     const vector<Tensor<T> *> &output)
{
}


template<typename T>
void ConvolutionLayer<T>::BackwardGPU(const vector<Tensor<T> *> &input,
                                      const vector<Tensor<T> *> &output)
{
}

template void ConvolutionLayer<float>::ForwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
//template void ConvolutionLayer<double>::ForwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
template void ConvolutionLayer<float>::BackwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
//template void ConvolutionLayer<double>::BackwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
}