#include "dropout_layer.hpp"

namespace z {

template<typename T>
void DropoutLayer<T>::ForwardGPU(const vector<container_type *> &input,
                                 const vector<container_type *> &output)
{
}


template<typename T>
void DropoutLayer<T>::BackwardGPU(const vector<container_type *> &input,
                                  const vector<container_type *> &output)
{
}

template void DropoutLayer<float>::ForwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
template void DropoutLayer<double>::ForwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
template void DropoutLayer<float>::BackwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
template void DropoutLayer<double>::BackwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
}