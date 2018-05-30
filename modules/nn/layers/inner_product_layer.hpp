#include "math/math_op.h"

namespace alchemy {

template <typename Device, typename T>
void InnerProductLayer<Device, T>::ForwardCPU(const vector<container *>& input,
                                      const vector<container *>& output)
{
    auto input_data = input[0]->data_cptr();
    auto weight = weights_->data_cptr();
    auto output_data = output[0]->mutable_data_cptr();

    // w * x
    matrix_mul(CblasNoTrans, CblasTrans,
               M_, N_, K_,
               (T)1, input_data, weight,
               (T)0, output_data);

    // output_data += bias
    matrix_mul(CblasNoTrans, CblasNoTrans,
               M_, N_, 1,
               (T)1, biasmer_.cptr(), biases_->data_cptr(),
               (T)1, output_data);
}

template <typename Device, typename T>
void InnerProductLayer<Device, T>::BackwardCPU(const vector<container *>& input,
                                       const vector<container *>& output)
{
    // 向前传递误差
    matrix_mul(CblasNoTrans, CblasNoTrans,
               M_, K_, N_,
               (T)1, output[0]->diff_cptr(), weights_->data_cptr(),
               (T)0, input[0]->mutable_diff_cptr());


    // 计算参数的更新值
    // weights
    matrix_mul(CblasTrans, CblasNoTrans,
               N_, K_, M_,
               (T)1, output[0]->diff_cptr(), input[0]->data_cptr(),
               (T)0, weights_->mutable_diff_cptr());

    // biases
    matvec_mul(CblasTrans, M_, N_,
               (T)1, output[0]->diff_cptr(), biasmer_.cptr(),
               (T)0, biases_->mutable_diff_cptr());
}
}
