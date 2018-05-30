#include "math/math_op.h"

namespace alchemy {

template <typename Device, typename T>
void SoftmaxLossLayer<Device, T>::setup(const vector<container *> &input,
                                const vector<container *> &output)
{
    LOG_IF(FATAL, input.size() < 2) << "input size: " << input.size();

    softmax_layer_ = shared_ptr<Layer<Device, T>>(
            new SoftmaxLayer<Device, T>(
                    LayerParameter()
                            .name("<<softmax_loss: softmax>>")
                            .type(SOFTMAX_LAYER)
                            .softmax_param(
                                    SoftmaxParameter()
                            )
            ));

    softmax_output_.push_back(shared_ptr<Blob<Device, T>>(new Blob<Device, T>()));
    softmax_layer_->setup(input, { softmax_output_[0].get() });

    output[0]->reset({ 1 });
}

template <typename Device, typename T>
void SoftmaxLossLayer<Device, T>::ForwardCPU(const vector<container *> &input,
                                     const vector<container *> &output)
{
    softmax_layer_->Forward(input, { softmax_output_[0].get() });

    //TODO: loss
}

template <typename Device, typename T>
void SoftmaxLossLayer<Device, T>::BackwardCPU(const vector<container *> &input,
                                      const vector<container *> &output)
{
    auto count = input[0]->size();
    auto label_data = input[1]->data_cptr();
    auto input_data = input[0]->data_cptr();
    auto input_diff = input[0]->mutable_diff_cptr();

    vector_sub(count, input_data, label_data, input_diff);
    vector_scal(count, (T)1.0/input[0]->shape(0), input_diff);
}
}