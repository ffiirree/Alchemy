#include "softmax_layer.h"
#include <glog/logging.h>
#include "math/math_op.h"

namespace alchemy {

template<typename T>
void SoftmaxLayer<T>::ForwardGPU(const vector<Tensor<T> *> &input,
                                 const vector<Tensor<T> *> &output)
{
    const auto count = input[0]->count();
    auto input_data = input[0]->gpu_data();
    auto output_data = output[0]->gpu_data();

    vector_copy_gpu(count, input_data, output_data);

    //TODO: numerical issues
    // exp
    vector_exp_gpu(count, output_data, output_data);
    // sum
    matrix_mul_gpu(CblasNoTrans, CblasNoTrans,
                   input[0]->shape(0), input[0]->shape(2), input[0]->shape(2),
                   (T)1., output_data, sum_multer_.gpu_data(),
                   (T)0., sum_.gpu_data());
    // div
    vector_div_gpu(count, output_data, sum_.gpu_data(), output_data);
}

template<typename T>
void SoftmaxLayer<T>::BackwardGPU(const vector<Tensor<T> *> &input,
                                  const vector<Tensor<T> *> &output)
{
    LOG(FATAL) << "Not implement!";
}

template void SoftmaxLayer<float>::ForwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
template void SoftmaxLayer<double>::ForwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
template void SoftmaxLayer<float>::BackwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
template void SoftmaxLayer<double>::BackwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
}