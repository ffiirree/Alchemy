#include "cudnn_conv_layer.h"

namespace alchemy {

template<typename T>
void CuDNNConvolutionLayer<T>::ForwardGPU(const vector<Blob<T> *> &input,
                                          const vector<Blob<T> *> &output)
{
    const T one = 1.0, zero = 0.0;

    for(size_t i = 0; i < output.size(); ++i) {
        // y = x * w + b

        // y = x * w
        CUDNN_CHECK(cudnnConvolutionForward(cudnn_,
                                            &one,
                                            input_descriptors_[i],
                                            input[i]->data_gptr(),
                                            kernel_descriptor_,
                                            this->kernel_->data_gptr(),
                                            conv_descriptors_[i],
                                            fwd_algorithms_[i],
                                            workspace_,
                                            workspace_fwd_sizes_[i],
                                            &zero,
                                            output_descriptors_[i],
                                            output[i]->mutable_data_gptr()));

        // + b
        CUDNN_CHECK(cudnnAddTensor(cudnn_,
                                   &one,
                                   bias_descriptor_,
                                   this->bias_->data_gptr(),
                                   &one,
                                   output_descriptors_[i],
                                   output[i]->mutable_data_gptr()));
    }
}


template<typename T>
void CuDNNConvolutionLayer<T>::BackwardGPU(const vector<Blob<T> *> &input,
                                           const vector<Blob<T> *> &output)
{
    const T one = 1.0, zero = 0.0;

    for(size_t i = 0; i < output.size(); ++i) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(cudnn_,
                                                 &one,
                                                 output_descriptors_[i],
                                                 output[i]->diff_gptr(),
                                                 &zero,
                                                 bias_descriptor_,
                                                 this->bias_->mutable_diff_gptr()));

        CUDNN_CHECK(cudnnConvolutionBackwardFilter(cudnn_,
                                                   &one,
                                                   input_descriptors_[i],
                                                   input[i]->data_gptr(),
                                                   output_descriptors_[i],
                                                   output[i]->diff_gptr(),
                                                   conv_descriptors_[i],
                                                   bwd_filter_algorithms_[i],
                                                   workspace_,
                                                   workspace_bwd_filter_sizes_[i],
                                                   &zero,
                                                   kernel_descriptor_,
                                                   this->kernel_->mutable_diff_gptr()));

        CUDNN_CHECK(cudnnConvolutionBackwardData(cudnn_,
                                                 &one,
                                                 kernel_descriptor_,
                                                 this->kernel_->data_gptr(),
                                                 output_descriptors_[i],
                                                 output[i]->diff_gptr(),
                                                 conv_descriptors_[i],
                                                 bwd_data_algorithms_[i],
                                                 workspace_,
                                                 workspace_bwd_data_sizes_[i],
                                                 &zero,
                                                 input_descriptors_[i],
                                                 input[i]->mutable_diff_gptr()));
    }
}

template void CuDNNConvolutionLayer<float>::ForwardGPU(const vector<Blob<float> *> &input, const vector<Blob<float> *> &output);
template void CuDNNConvolutionLayer<double>::ForwardGPU(const vector<Blob<double> *> &input, const vector<Blob<double> *> &output);
template void CuDNNConvolutionLayer<float>::BackwardGPU(const vector<Blob<float> *> &input, const vector<Blob<float> *> &output);
template void CuDNNConvolutionLayer<double>::BackwardGPU(const vector<Blob<double> *> &input, const vector<Blob<double> *> &output);
}