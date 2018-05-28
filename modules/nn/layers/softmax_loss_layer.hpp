#include "math/math_op.h"

namespace alchemy {

template<typename T>
void SoftmaxLossLayer<T>::setup(const vector<Blob<T> *> &input,
                                const vector<Blob<T> *> &output)
{
    LOG_IF(FATAL, input.size() < 2) << "input size: " << input.size();

    softmax_layer_ = shared_ptr<Layer<T>>(
            new SoftmaxLayer<T>(
                    LayerParameter()
                            .name("<<softmax_loss: softmax>>")
                            .type(SOFTMAX_LAYER)
                            .softmax_param(
                                    SoftmaxParameter()
                            )
            ));

    softmax_output_.push_back(shared_ptr<Blob<T>>(new Blob<T>()));
    softmax_layer_->setup(input, { softmax_output_[0].get() });

    output[0]->reshape({ 1 });
}

template<typename T>
void SoftmaxLossLayer<T>::ForwardCPU(const vector<Blob<T> *> &input,
                                     const vector<Blob<T> *> &output)
{
    softmax_layer_->Forward(input, { softmax_output_[0].get() });

    //TODO: loss
}

template<typename T>
void SoftmaxLossLayer<T>::BackwardCPU(const vector<Blob<T> *> &input,
                                      const vector<Blob<T> *> &output)
{
    auto count = input[0]->count();
    auto label_data = input[1]->data_cptr();
    auto input_data = input[0]->data_cptr();
    auto input_diff = input[0]->mutable_diff_cptr();

    vector_sub(count, input_data, label_data, input_diff);
    vector_scal(count, (T)1.0/input[0]->shape(0), input_diff);
}
}