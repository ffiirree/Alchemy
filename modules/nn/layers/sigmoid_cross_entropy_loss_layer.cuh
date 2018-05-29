#include <math/math_op.h>

namespace alchemy {

template <typename Device, typename T>
void SigmoidCrossEntropyLossLayer<Device, T>::ForwardGPU(const vector<container *> &input,
                                                 const vector<container *> &output)
{
    // computes the sigmoid outputs.
    sigmoid_layers_->Forward(input, { sigmoid_output_[0].get() });

    //TODO: loss
}

template <typename Device, typename T>
void SigmoidCrossEntropyLossLayer<Device, T>::BackwardGPU(const vector<container *> &input,
                                                  const vector<container *> &output)
{
    auto sigmoid_output = sigmoid_output_[0]->data_gptr();
    auto target = input[1]->data_gptr();
    auto count = sigmoid_output_[0]->size();
    auto diff = input[0]->mutable_diff_gptr();

    vector_sub_gpu(count, sigmoid_output, target, diff);
    vector_scal_gpu(count, (T)1.0/input[0]->shape(0), diff);
}
}