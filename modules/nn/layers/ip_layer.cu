#include "ip_layer.h"
#include "math/math_op.h"

namespace alchemy {

template<typename T>
void InnerProductLayer<T>::ForwardGPU(const vector<Tensor<T> *> &input,
                                      const vector<Tensor<T> *> &output)
{
    auto input_data = input[0]->gpu_data();
    auto weight = weights_->gpu_data();
    auto output_data = output[0]->gpu_data();

    // w * x
    matrix_mul_gpu(CblasNoTrans, CblasTrans,
                   M_, N_, K_,
                   (T)1, input_data, weight,
                   (T)0, output_data);

    // output_data += bias
    matrix_mul_gpu(CblasNoTrans, CblasNoTrans,
                   M_, N_, 1,
                   (T)1, biasmer_.gpu_data(), biases_->gpu_data(),
                   (T)1, output_data);
}

template<typename T>
void InnerProductLayer<T>::BackwardGPU(const vector<Tensor<T> *> &input,
                                       const vector<Tensor<T> *> &output)
{
    // 向前传递误差
    matrix_mul_gpu(CblasNoTrans, CblasNoTrans,
                   M_, K_, N_,
                   (T)1, output[0]->gpu_diff(), weights_->gpu_data(),
                   (T)0, input[0]->gpu_diff());


    // 计算参数的更新值
    // weights
    matrix_mul_gpu(CblasTrans, CblasNoTrans,
                   N_, K_, M_,
                   (T)1, output[0]->gpu_diff(), input[0]->gpu_data(),
                   (T)0, weights_->gpu_diff());

    // biases
    matvec_mul_gpu(CblasTrans, M_, N_,
                   (T)1, output[0]->gpu_diff(), biasmer_.gpu_data(),
                   (T)0, biases_->gpu_diff());
}

template void InnerProductLayer<float>::ForwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
template void InnerProductLayer<double>::ForwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
template void InnerProductLayer<float>::BackwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
template void InnerProductLayer<double>::BackwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
}