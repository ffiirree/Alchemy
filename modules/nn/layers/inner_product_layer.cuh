#ifndef ALCHEMY_NN_LAYERS_INNER_PRODUCT_CUH
#define ALCHEMY_NN_LAYERS_INNER_PRODUCT_CUH

#include "math/math_op.h"

namespace alchemy {

template <typename Device, typename T>
void InnerProductLayer<Device, T>::ForwardGPU(const vector<container *> &input,
                                      const vector<container *> &output)
{
    auto input_data = input[0]->data_gptr();
    auto weight = weights_->data_gptr();
    auto output_data = output[0]->mutable_data_gptr();

    // w * x
    matrix_mul_gpu(CblasNoTrans, CblasTrans,
                   M_, N_, K_,
                   (T)1, input_data, weight,
                   (T)0, output_data);

    // output_data += bias
    matrix_mul_gpu(CblasNoTrans, CblasNoTrans,
                   M_, N_, 1,
                   (T)1, biasmer_.gptr(), biases_->data_gptr(),
                   (T)1, output_data);
}

template <typename Device, typename T>
void InnerProductLayer<Device, T>::BackwardGPU(const vector<container *> &input,
                                       const vector<container *> &output)
{
    // 向前传递误差
    matrix_mul_gpu(CblasNoTrans, CblasNoTrans,
                   M_, K_, N_,
                   (T)1, output[0]->diff_gptr(), weights_->data_gptr(),
                   (T)0, input[0]->mutable_diff_gptr());


    // 计算参数的更新值
    // weights
    matrix_mul_gpu(CblasTrans, CblasNoTrans,
                   N_, K_, M_,
                   (T)1, output[0]->diff_gptr(), input[0]->data_gptr(),
                   (T)0, weights_->mutable_diff_gptr());

    // biases
    matvec_mul_gpu(CblasTrans, M_, N_,
                   (T)1, output[0]->diff_gptr(), biasmer_.gptr(),
                   (T)0, biases_->mutable_diff_gptr());
}
}
#endif// !ALCHEMY_NN_LAYERS_INNER_PRODUCT_CUH