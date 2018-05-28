#include "math/math_op.h"

namespace alchemy {

template<typename T>
void SoftmaxLossLayer<T>::ForwardGPU(const vector<Blob<T> *> &input,
                                     const vector<Blob<T> *> &output)
{
    softmax_layer_->Forward(input, { softmax_output_[0].get() });

    //TODO: loss
}

template<typename T>
void SoftmaxLossLayer<T>::BackwardGPU(const vector<Blob<T> *> &input,
                                      const vector<Blob<T> *> &output)
{
    auto count = input[0]->count();
    auto label_data = input[1]->data_gptr();
    auto input_data = input[0]->data_gptr();
    auto input_diff = input[0]->mutable_diff_gptr();

    vector_sub_gpu(count, input_data, label_data, input_diff);
    vector_scal_gpu(count, (T)1.0/input[0]->shape(0), input_diff);
}
}