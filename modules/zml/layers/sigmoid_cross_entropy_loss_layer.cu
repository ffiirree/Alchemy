#include <zml/util/math_op.hpp>
#include "sigmoid_cross_entropy_loss_layer.hpp"

namespace z {

template<typename T>
void SigmoidCrossEntropyLossLayer<T>::ForwardGPU(const vector<container_type *> &input,
                                                 const vector<container_type *> &output)
{
    // computes the sigmoid outputs.
    sigmoid_layers_->Forward(input, { sigmoid_output_[0].get() });

    //TODO: loss
}

template<typename T>
void SigmoidCrossEntropyLossLayer<T>::BackwardGPU(const vector<container_type *> &input,
                                                  const vector<container_type *> &output)
{
    auto sigmoid_output = sigmoid_output_[0]->gpu_data();
    auto target = input[1]->gpu_data();
    auto count = sigmoid_output_[0]->count();
    auto diff = input[0]->gpu_diff();

    vector_sub_gpu(count, sigmoid_output, target, diff);
    vector_scal_gpu(count, (T)1.0/input[0]->shape(0), diff);
}

template void SigmoidCrossEntropyLossLayer<float>::ForwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
template void SigmoidCrossEntropyLossLayer<double>::ForwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
template void SigmoidCrossEntropyLossLayer<float>::BackwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
template void SigmoidCrossEntropyLossLayer<double>::BackwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
}