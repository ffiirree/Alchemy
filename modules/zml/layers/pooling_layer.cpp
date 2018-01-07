#include <glog/logging.h>
#include "pooling_layer.hpp"

namespace z {

template<typename T>
void PoolingLayer<T>::setup(const vector<container_type *> &input,
                            const vector<container_type *> &output)
{
    assert((size_t)input[0]->shape(2) > pooling_param_.kernel_size());
    assert((size_t)input[0]->shape(3) > pooling_param_.kernel_size());

    auto ksize = pooling_param_.kernel_size();
    auto num_in = input[0]->shape(0);
    auto chs_in = input[0]->shape(1);
    auto row_in = input[0]->shape(2);
    auto col_in = input[0]->shape(3);

    auto row_out = static_cast<int>((row_in - ksize) / pooling_param_.stride() + 1);
    auto col_out = static_cast<int>((col_in - ksize) / pooling_param_.stride() + 1);

    output[0]->reshape({ num_in, chs_in, row_out, col_out });

    LOG(INFO) << "Pooling Layer: { out: " << output[0]->shape() << " }";
}

template<typename T>
void PoolingLayer<T>::ForwardCPU(const vector<container_type *> &input,
                                 const vector<container_type *> &output)
{

}

template<typename T>
void PoolingLayer<T>::BackwardCPU(const vector<container_type *> &input,
                                  const vector<container_type *> &output)
{

}

template class PoolingLayer<float>;
template class PoolingLayer<double>;
}