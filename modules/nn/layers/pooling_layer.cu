#include "pooling_layer.h"

namespace alchemy {

template<typename T>
void PoolingLayer<T>::ForwardGPU(const vector<Blob<T> *> &input,
                                 const vector<Blob<T> *> &output)
{

}

template<typename T>
void PoolingLayer<T>::BackwardGPU(const vector<Blob<T> *> &input,
                                  const vector<Blob<T> *> &output)
{

}

template void PoolingLayer<float>::ForwardGPU(const vector<Blob<float> *> &input, const vector<Blob<float> *> &output);
template void PoolingLayer<double>::ForwardGPU(const vector<Blob<double> *> &input, const vector<Blob<double> *> &output);
template void PoolingLayer<float>::BackwardGPU(const vector<Blob<float> *> &input, const vector<Blob<float> *> &output);
template void PoolingLayer<double>::BackwardGPU(const vector<Blob<double> *> &input, const vector<Blob<double> *> &output);
}