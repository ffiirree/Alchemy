#include <glog/logging.h>
#include <nnpack.h>
#include "math/math_op.h"
#include "nn/layer_factory.h"

namespace alchemy {

template <typename Device, typename T>
void ConvolutionLayer<Device, T>::setup(const vector<container *> &input,
                                const vector<container *> &output)
{
    assert(input[0]->shape(2) > conv_param_.kernel_size());
    assert(input[0]->shape(3) > conv_param_.kernel_size());

    if(nnp_initialize() != nnp_status_success) {
        LOG(INFO) << "NNPACK failed to initialize!";
    }

    auto ksize = conv_param_.kernel_size();
    auto num_in = input[0]->shape(0);
    auto chs_in = input[0]->shape(1);
    auto row_in = input[0]->shape(2);
    auto col_in = input[0]->shape(3);

    auto chs_out = conv_param_.output_size();
    auto row_out = (row_in - ksize) / conv_param_.stride() + 1;
    auto col_out = (col_in - ksize) / conv_param_.stride() + 1;

    output[0]->reset({ num_in, chs_out, row_out, col_out });

    if(this->learnable_params_.empty()) {

        kernel_->reset({ chs_in, chs_out, ksize, ksize });
        bias_->reset({ chs_out }); // { 1, chs_out, 1, 1 }

        Filler<Device, T>::fill(kernel_->data(), conv_param_.weight_filler());
        Filler<Device, T>::fill(bias_->data(), conv_param_.bias_filler());

        this->learnable_params_.resize(2);
        this->learnable_params_[0] = std::make_tuple(kernel_, conv_param_.wlr(), conv_param_.weight_decay()/input[0]->shape(0));
        this->learnable_params_[1] = std::make_tuple(bias_, conv_param_.blr(), 0.0);

        biaser_.reset({ 1, output[0]->size(2, 4) });
//        vector_set(biaser_.size(), (T)1.0, biaser_.mutable_cptr());
        Filler<Device, T>::constant_fill(biaser_.size(), biaser_.mutable_cptr(), (T)1.0);
    }
}

template <typename Device, typename T>
void ConvolutionLayer<Device, T>::ForwardCPU(const vector<container *> &input,
                                     const vector<container *> &output)
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

template <typename Device, typename T>
void ConvolutionLayer<Device, T>::BackwardCPU(const vector<container *> &input,
                                      const vector<container *> &output)
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
    Filler<Device, T>::constant_fill(bias_->size(), bias_->mutable_data_cptr(), (T)0.0);
//    vector_set(bias_->size(), (T)0.0, bias_->mutable_diff_cptr());

    auto output_diff = output[0]->diff_cptr();
    auto step = output[0]->size(1, 4);
    for(size_t i = 0; i < batch_size; ++i) {
        matvec_mul<Device>(CblasNoTrans,
                   output[0]->shape(1), output[0]->size(2, 4),
                   (T)1.0, output_diff + i * step, biaser_.ptr(),
                   (T)1.0, bias_->mutable_diff_ptr());
    }
}
}
