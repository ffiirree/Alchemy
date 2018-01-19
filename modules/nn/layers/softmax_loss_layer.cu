#include "softmax_loss_layer.h"
#include "math/math_op.h"

namespace alchemy {

template<typename T>
void SoftmaxLossLayer<T>::ForwardGPU(const vector<Tensor<T> *> &input,
                                     const vector<Tensor<T> *> &output)
{
    softmax_layer_->Forward(input, { softmax_output_[0].get() });

    //TODO: loss
}

template<typename T>
void SoftmaxLossLayer<T>::BackwardGPU(const vector<Tensor<T> *> &input,
                                      const vector<Tensor<T> *> &output)
{
    const auto count = input[0]->count();
    const auto label_data = input[1]->gpu_data();
    const auto input_data = input[0]->gpu_data();
    auto input_diff = input[0]->gpu_diff();

    vector_sub_gpu(count, input_data, label_data, input_diff);
    vector_scal_gpu(count, (T)1.0/input[0]->shape(0), input_diff);
}

template void SoftmaxLossLayer<float>::ForwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
template void SoftmaxLossLayer<double>::ForwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
template void SoftmaxLossLayer<float>::BackwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
template void SoftmaxLossLayer<double>::BackwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
}