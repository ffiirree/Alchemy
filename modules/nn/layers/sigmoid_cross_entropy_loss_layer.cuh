#include <math/math_op.h>

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
    auto diff = input[0]->mutable_diff_gptr();

    vector_sub_gpu(count, sigmoid_output, target, diff);
    vector_scal_gpu(count, (T)1.0/input[0]->shape(0), diff);
}
}