namespace alchemy {

template <typename Device, typename T>
void CuDNNConvolutionLayer<Device, T>::ForwardGPU(const vector<container *> &input,
                                          const vector<container *> &output)
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


template<typename Device, typename T>
void CuDNNConvolutionLayer<Device, T>::BackwardGPU(const vector<container *> &input,
                                           const vector<container *> &output)
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
}