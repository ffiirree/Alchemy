#ifndef ALCHEMY_NN_LAYERS_CUDNN_CONV_LAYER_H
#define ALCHEMY_NN_LAYERS_CUDNN_CONV_LAYER_H

#include <cudnn.h>
#include "conv_layer.h"

namespace alchemy {

namespace cudnn {
template <typename T> struct DataType;
template <> struct DataType<float> {
    static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
};
template <> struct DataType<double> {
    static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
};
}

template <typename Device, typename T>
class CuDNNConvolutionLayer : public ConvolutionLayer<Device, T> {
public:
    using container = Blob<Device, T>;
    
    explicit CuDNNConvolutionLayer() = default;
    explicit CuDNNConvolutionLayer(const LayerParameter&param)
            : ConvolutionLayer<Device, T>(param) { }
    virtual ~CuDNNConvolutionLayer();

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void ForwardGPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardGPU(const vector<container *>& input, const vector<container *>& output) override;

private:
    cudnnHandle_t cudnn_{};
    cudnnConvolutionFwdAlgo_t * fwd_algorithms_ = nullptr;
    cudnnConvolutionBwdFilterAlgo_t * bwd_filter_algorithms_ = nullptr;
    cudnnConvolutionBwdDataAlgo_t * bwd_data_algorithms_ = nullptr;

    vector<cudnnTensorDescriptor_t> input_descriptors_, output_descriptors_;
    cudnnTensorDescriptor_t bias_descriptor_{};
    cudnnFilterDescriptor_t kernel_descriptor_{};
    vector<cudnnConvolutionDescriptor_t> conv_descriptors_;

    size_t * workspace_fwd_sizes_ = nullptr;
    size_t * workspace_bwd_data_sizes_ = nullptr;
    size_t * workspace_bwd_filter_sizes_ = nullptr;
    void * workspace_ = nullptr;
};
}

#include "cudnn_conv_layer.hpp"
#ifdef __CUDACC__
#include "cudnn_conv_layer.cuh"
#endif //! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_CUDNN_CONV_LAYER_H
