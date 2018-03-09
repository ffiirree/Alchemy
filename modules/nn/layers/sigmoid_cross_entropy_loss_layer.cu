#include <math/math_op.h>
#include "sigmoid_cross_entropy_loss_layer.h"

namespace alchemy {

template<typename T>
void SigmoidCrossEntropyLossLayer<T>::ForwardGPU(const vector<Blob<T> *> &input,
                                                 const vector<Blob<T> *> &output)
{
    // computes the sigmoid outputs.
    sigmoid_layers_->Forward(input, { sigmoid_output_[0].get() });

    //TODO: loss
}

template<typename T>
void SigmoidCrossEntropyLossLayer<T>::BackwardGPU(const vector<Blob<T> *> &input,
                                                  const vector<Blob<T> *> &output)
{
    auto sigmoid_output = sigmoid_output_[0]->data_gptr();
    auto target = input[1]->data_gptr();
    auto count = sigmoid_output_[0]->count();
    auto diff = input[0]->diff_gptr();

    vector_sub_gpu(count, sigmoid_output, target, diff);
    vector_scal_gpu(count, (T)1.0/input[0]->shape(0), diff);
}

template void SigmoidCrossEntropyLossLayer<float>::ForwardGPU(const vector<Blob<float> *> &input, const vector<Blob<float> *> &output);
template void SigmoidCrossEntropyLossLayer<double>::ForwardGPU(const vector<Blob<double> *> &input, const vector<Blob<double> *> &output);
template void SigmoidCrossEntropyLossLayer<float>::BackwardGPU(const vector<Blob<float> *> &input, const vector<Blob<float> *> &output);
template void SigmoidCrossEntropyLossLayer<double>::BackwardGPU(const vector<Blob<double> *> &input, const vector<Blob<double> *> &output);
}