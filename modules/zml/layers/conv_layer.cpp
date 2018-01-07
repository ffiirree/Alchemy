#include <glog/logging.h>
#include "conv_layer.hpp"

namespace z {

template<typename T>
void ConvolutionLayer<T>::setup(const vector<container_type *> &input,
                                const vector<container_type *> &output)
{
    assert((size_t)input[0]->shape(2) > conv_param_.kernel_size());
    assert((size_t)input[0]->shape(3) > conv_param_.kernel_size());

    auto ksize = conv_param_.kernel_size();
    int num_in = input[0]->shape(0);
    int row_in = input[0]->shape(2);
    int col_in = input[0]->shape(3);

    auto chs_out = static_cast<int>(conv_param_.output_size());
    auto row_out = static_cast<int>((row_in - ksize) / conv_param_.stride() + 1);
    auto col_out = static_cast<int>((col_in - ksize) / conv_param_.stride() + 1);

    output[0]->reshape({ num_in, chs_out, row_out, col_out });

    LOG(INFO) << "Conv Layer: { out: " << output[0]->shape() << " }, "
              << "{ kernel: ("
              << conv_param_.output_size() << ", "
              << ksize << ", "
              << ksize << ") }";
}

template<typename T>
void ConvolutionLayer<T>::ForwardCPU(const vector<container_type *> &input,
                                     const vector<container_type *> &output)
{

}

template<typename T>
void ConvolutionLayer<T>::BackwardCPU(const vector<container_type *> &input,
                                      const vector<container_type *> &output)
{

}

template class ConvolutionLayer<float>;
template class ConvolutionLayer<double>;
}
