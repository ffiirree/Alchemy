#include "conv_layer.hpp"

namespace z {

template<typename T>
void ConvolutionLayer<T>::ForwardGPU(const vector<container_type *> &input,
                                     const vector<container_type *> &output)
{
}


template<typename T>
void ConvolutionLayer<T>::BackwardGPU(const vector<container_type *> &input,
                                      const vector<container_type *> &output)
{
}

template void ConvolutionLayer<float>::ForwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
//template void ConvolutionLayer<double>::ForwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
template void ConvolutionLayer<float>::BackwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
//template void ConvolutionLayer<double>::BackwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
}