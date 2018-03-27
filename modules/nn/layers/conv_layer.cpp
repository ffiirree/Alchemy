#include "conv_layer.h"
#include <glog/logging.h>
#include <nnpack.h>
#include "math/math_op.h"
#include "nn/layer_factory.h"

namespace alchemy {

template<typename T>
void ConvolutionLayer<T>::setup(const vector<Blob<T> *> &input,
                                const vector<Blob<T> *> &output)
{
    LOG(INFO) << "Setting up: " << this->param_.name();
    LOG(INFO) << "input  #0: "  << input[0]->shape();

    assert((size_t)input[0]->shape(2) > conv_param_.kernel_size());
    assert((size_t)input[0]->shape(3) > conv_param_.kernel_size());

    if(nnp_initialize() != nnp_status_success) {
        LOG(INFO) << "NNPACK failed to initialize!";
    }

    auto ksize = conv_param_.kernel_size();
    int num_in = input[0]->shape(0);
    int chs_in = input[0]->shape(1);
    int row_in = input[0]->shape(2);
    int col_in = input[0]->shape(3);

    auto chs_out = static_cast<int>(conv_param_.output_size());
    auto row_out = static_cast<int>((row_in - ksize) / conv_param_.stride() + 1);
    auto col_out = static_cast<int>((col_in - ksize) / conv_param_.stride() + 1);

    output[0]->reshape({ num_in, chs_out, row_out, col_out });
    LOG(INFO) << "output #0: "  << output[0]->shape();

    if(this->learnable_params_.empty()) {
        kernel_ = LayerFactory<T>::GetSharedParam(this->param_.name(), 0);
        bias_ = LayerFactory<T>::GetSharedParam(this->param_.name(), 1);

        kernel_->reshape({ chs_in, chs_out, (int)ksize, (int)ksize });
        bias_->reshape({ chs_out }); // { 1, chs_out, 1, 1 }

        Filler<T>::fill(kernel_->data(), conv_param_.weight_filler());
        Filler<T>::fill(bias_->data(), conv_param_.bias_filler());

        this->learnable_params_.resize(2);
        this->learnable_params_[0] = std::make_tuple(kernel_, conv_param_.wlr(), conv_param_.weight_decay()/input[0]->shape(0));
        this->learnable_params_[1] = std::make_tuple(bias_, conv_param_.blr(), 0.0);

        biaser_.reshape({ 1, output[0]->count(2, 4) });
        vector_set(biaser_.count(), (T)1.0, biaser_.mutable_cptr());
    }
}

template<typename T>
void ConvolutionLayer<T>::ForwardCPU(const vector<Blob<T> *> &input,
                                     const vector<Blob<T> *> &output)
{
    auto input_data = input[0]->data_cptr();
    auto output_data = output[0]->mutable_data_cptr();
    auto kernel = kernel_->data_cptr();
    auto bias = bias_->data_cptr();
    const size_t batch_size = input[0]->shape(0);
    const size_t chs_in = input[0]->shape(1);
    const size_t chs_out = output[0]->shape(1);
    const nnp_padding padding_in = { 0, 0, 0, 0 };
    const nnp_size input_size = { (size_t)input[0]->shape(3), (size_t)input[0]->shape(2) };
    const nnp_size kernel_size = { conv_param_.kernel_size(), conv_param_.kernel_size() };

    nnp_convolution_output(nnp_convolution_algorithm_auto,
                           batch_size,
                           chs_in, chs_out,
                           input_size, padding_in, kernel_size,
                           input_data, kernel, bias, output_data,
                           nullptr, nullptr);

}

template<typename T>
void ConvolutionLayer<T>::BackwardCPU(const vector<Blob<T> *> &input,
                                      const vector<Blob<T> *> &output)
{
    auto kernel = kernel_->data_cptr();
    const size_t batch_size = input[0]->shape(0);
    const size_t chs_in = input[0]->shape(1);
    const size_t chs_out = output[0]->shape(1);
    const nnp_padding padding_in = { 0, 0, 0, 0 };
    const nnp_size input_size = { (size_t)input[0]->shape(2), (size_t)input[0]->shape(3) };
    const nnp_size kernel_size = { conv_param_.kernel_size(), conv_param_.kernel_size() };

    nnp_convolution_input_gradient(nnp_convolution_algorithm_auto,
                                   batch_size,
                                   chs_in, chs_out,
                                   input_size, padding_in, kernel_size,
                                   output[0]->diff_cptr(),
                                   kernel,
                                   input[0]->mutable_diff_cptr(),
                                   nullptr, nullptr);

    nnp_convolution_kernel_gradient(nnp_convolution_algorithm_auto,
                                    batch_size,
                                    chs_in, chs_out,
                                    input_size, padding_in, kernel_size,
                                    input[0]->data_cptr(),
                                    output[0]->diff_cptr(),
                                    kernel_->mutable_diff_cptr(),
                                    nullptr,
                                    nullptr);

    //
    vector_set(bias_->count(), (T)0.0, bias_->mutable_diff_cptr());

    auto output_diff = output[0]->diff_cptr();
    auto step = output[0]->count(1, 4);
    for(size_t i = 0; i < batch_size; ++i) {
        matvec_mul(CblasNoTrans,
                   output[0]->shape(1), output[0]->count(2, 4),
                   (T)1.0, output_diff + i * step, biaser_.cptr(),
                   (T)1.0, bias_->mutable_diff_cptr());
    }
}

template class ConvolutionLayer<float>;
//template class ConvolutionLayer<double>;
}
