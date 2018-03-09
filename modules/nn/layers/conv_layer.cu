#include "conv_layer.h"

namespace alchemy {

template<typename T>
void ConvolutionLayer<T>::ForwardGPU(const vector<Blob<T> *> &input,
                                     const vector<Blob<T> *> &output)
{
}


template<typename T>
void ConvolutionLayer<T>::BackwardGPU(const vector<Blob<T> *> &input,
                                      const vector<Blob<T> *> &output)
{
}

template void ConvolutionLayer<float>::ForwardGPU(const vector<Blob<float> *> &input, const vector<Blob<float> *> &output);
//template void ConvolutionLayer<double>::ForwardGPU(const vector<Blob<double> *> &input, const vector<Blob<double> *> &output);
template void ConvolutionLayer<float>::BackwardGPU(const vector<Blob<float> *> &input, const vector<Blob<float> *> &output);
//template void ConvolutionLayer<double>::BackwardGPU(const vector<Blob<double> *> &input, const vector<Blob<double> *> &output);
}