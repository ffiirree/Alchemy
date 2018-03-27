#include "cudnn_conv_layer.h"

namespace alchemy {

template<typename T>
void CuDNNConvolutionLayer<T>::setup(const vector<Blob<T> *> &input,
                                     const vector<Blob<T> *> &output)
{
    ConvolutionLayer<T>::setup(input, output);

    CUDNN_CHECK(cudnnCreate(&cudnn_));

    for(size_t i = 0; i < input.size(); ++i) {
        // input
        cudnnTensorDescriptor_t input_descriptor;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_descriptor));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_descriptor,
                                               CUDNN_TENSOR_NCHW,
                                               cudnn::DataType<T>::type,
                                               input[i]->shape(0),
                                               input[i]->shape(1),
                                               input[i]->shape(2),
                                               input[i]->shape(3)));
        input_descriptors_.push_back(input_descriptor);

        // output
        cudnnTensorDescriptor_t output_descriptor;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_descriptor));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_descriptor,
                                               CUDNN_TENSOR_NCHW,
                                               cudnn::DataType<T>::type,
                                               output[i]->shape(0),
                                               output[i]->shape(1),
                                               output[i]->shape(2),
                                               output[i]->shape(3)));
        output_descriptors_.push_back(output_descriptor);

        // convolution
        cudnnConvolutionDescriptor_t conv_descriptor;
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_descriptor));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_descriptor,
                                                    0, 0,
                                                    this->conv_param_.stride(),
                                                    this->conv_param_.stride(),
                                                    1, 1,
                                                    CUDNN_CROSS_CORRELATION,
                                                    cudnn::DataType<T>::type));
        conv_descriptors_.push_back(conv_descriptor);
    }

    // filter
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&kernel_descriptor_));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(kernel_descriptor_,
                                           cudnn::DataType<T>::type,
                                           CUDNN_TENSOR_NCHW,
                                           this->conv_param_.output_size(),
                                           input[0]->shape(1),
                                           this->conv_param_.kernel_size(),
                                           this->conv_param_.kernel_size()));

    // bias
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_descriptor_));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_descriptor_,
                                           CUDNN_TENSOR_NCHW,
                                           cudnn::DataType<T>::type,
                                           1,  static_cast<int>(this->conv_param_.output_size()), 1, 1));

    // algorithm && workspace size
    fwd_algorithms_ = new cudnnConvolutionFwdAlgo_t[input.size()]();
    bwd_filter_algorithms_ = new cudnnConvolutionBwdFilterAlgo_t[input.size()]();
    bwd_data_algorithms_ = new cudnnConvolutionBwdDataAlgo_t[input.size()]();

    workspace_fwd_sizes_ = new size_t[input.size()]();
    workspace_bwd_filter_sizes_ = new size_t[input.size()]();
    workspace_bwd_data_sizes_ = new size_t[input.size()]();
    size_t workspace_fwd_size = 0, workspace_bwd_filter_size = 0, workspace_bwd_data_size = 0;
    size_t workspace_size = 0;
    for(size_t i = 0; i < input.size(); ++i) {

        // forward
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(cudnn_,
                                                        input_descriptors_[i],
                                                        kernel_descriptor_,
                                                        conv_descriptors_[i],
                                                        output_descriptors_[i],
                                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                        0,
                                                        &fwd_algorithms_[i]));

        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn_,
                                                            input_descriptors_[i],
                                                            kernel_descriptor_,
                                                            conv_descriptors_[i],
                                                            output_descriptors_[i],
                                                            fwd_algorithms_[i],
                                                            &workspace_fwd_sizes_[i]));
        workspace_fwd_size = std::max(workspace_fwd_size, workspace_fwd_sizes_[i]);

        // backward
        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_,
                                                               input_descriptors_[i],
                                                               output_descriptors_[i],
                                                               conv_descriptors_[i],
                                                               kernel_descriptor_,
                                                               CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                                               0,
                                                               &bwd_filter_algorithms_[i]));

        CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_,
                                                                   input_descriptors_[i],
                                                                   output_descriptors_[i],
                                                                   conv_descriptors_[i],
                                                                   kernel_descriptor_,
                                                                   bwd_filter_algorithms_[i],
                                                                   &workspace_bwd_filter_sizes_[i]));

        workspace_bwd_filter_size = std::max(workspace_bwd_filter_size, workspace_bwd_filter_sizes_[i]);
        CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_,
                                                             kernel_descriptor_,
                                                             output_descriptors_[i],
                                                             conv_descriptors_[i],
                                                             input_descriptors_[i],
                                                             CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                             0,
                                                             &bwd_data_algorithms_[0]));

        CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_,
                                                                 kernel_descriptor_,
                                                                 output_descriptors_[i],
                                                                 conv_descriptors_[i],
                                                                 input_descriptors_[i],
                                                                 bwd_data_algorithms_[i],
                                                                 &workspace_bwd_data_sizes_[i]));
        workspace_bwd_data_size = std::max(workspace_bwd_data_size, workspace_bwd_data_sizes_[i]);
    }

    workspace_size = std::max(workspace_bwd_data_size, workspace_bwd_filter_size);
    workspace_size = std::max(workspace_size, workspace_fwd_size);
    // allocate memory
    cudaMalloc(&workspace_, workspace_size);
}

template<typename T>
CuDNNConvolutionLayer<T>::~CuDNNConvolutionLayer()
{
    cudaFree(workspace_);
    delete workspace_fwd_sizes_;
    delete workspace_bwd_filter_sizes_;
    delete workspace_bwd_data_sizes_;
    delete fwd_algorithms_;
    delete bwd_filter_algorithms_;
    delete bwd_data_algorithms_;

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_descriptor_));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(kernel_descriptor_));

    for(size_t i = 0; i < input_descriptors_.size(); ++i) {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_descriptors_[i]));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_descriptors_[i]));
        CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_descriptors_[i]));
    }

    cudnnDestroy(cudnn_);
}

template class CuDNNConvolutionLayer<float>;
//template class CuDNNConvolutionLayer<double>;
}