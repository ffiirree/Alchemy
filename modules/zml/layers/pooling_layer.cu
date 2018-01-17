#include "pooling_layer.hpp"

namespace z {

template<typename T>
void PoolingLayer<T>::ForwardGPU(const vector<container_type *> &input,
                                 const vector<container_type *> &output)
{

}

template<typename T>
void PoolingLayer<T>::BackwardGPU(const vector<container_type *> &input,
                                  const vector<container_type *> &output)
{

}

template void PoolingLayer<float>::ForwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
template void PoolingLayer<double>::ForwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
template void PoolingLayer<float>::BackwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
template void PoolingLayer<double>::BackwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
}