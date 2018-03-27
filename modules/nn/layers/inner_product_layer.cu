#include "inner_product_layer.h"
#include "math/math_op.h"

namespace alchemy {

template<typename T>
void InnerProductLayer<T>::ForwardGPU(const vector<Blob<T> *> &input,
                                      const vector<Blob<T> *> &output)
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

template<typename T>
void InnerProductLayer<T>::BackwardGPU(const vector<Blob<T> *> &input,
                                       const vector<Blob<T> *> &output)
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

template void InnerProductLayer<float>::ForwardGPU(const vector<Blob<float> *> &input, const vector<Blob<float> *> &output);
template void InnerProductLayer<double>::ForwardGPU(const vector<Blob<double> *> &input, const vector<Blob<double> *> &output);
template void InnerProductLayer<float>::BackwardGPU(const vector<Blob<float> *> &input, const vector<Blob<float> *> &output);
template void InnerProductLayer<double>::BackwardGPU(const vector<Blob<double> *> &input, const vector<Blob<double> *> &output);
}