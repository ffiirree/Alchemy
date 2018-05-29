#include "math/math_op.h"

namespace alchemy {

template <typename Device, typename T>
void SigmoidCrossEntropyLossLayer<Device, T>::setup(const vector<container *> &input,
                                            const vector<container *> &output)
{
    LOG_IF(FATAL, input.size() < 2) << "input size: " << input.size();

    sigmoid_layers_ = shared_ptr<Layer<Device, T>>(
            new SigmoidLayer<Device, T>(
                    LayerParameter()
                            .name("<<sigmoid_cross_entropy_loss: sigmoid>>")
                            .type(SIGMOID_LAYER)
                            .sigmoid_param(
                                    SigmoidParameter()
                            )
            ));
    sigmoid_output_.push_back(shared_ptr<Blob<Device, T>>(new Blob<Device, T>()));
    sigmoid_layers_->setup(input, { sigmoid_output_[0].get() });

    output[0]->reshape({ 1 });
}


template <typename Device, typename T>
void SigmoidCrossEntropyLossLayer<Device, T>::ForwardCPU(const vector<container *> &input,
                                                 const vector<container *> &output)
{
    // computes the sigmoid outputs.
    sigmoid_layers_->Forward(input, { sigmoid_output_[0].get() });

    //TODO: loss
}

template <typename Device, typename T>
void  SigmoidCrossEntropyLossLayer<Device, T>::BackwardCPU(const vector<container *> &input,
                                                   const vector<container *> &output)
{
    auto sigmoid_output = sigmoid_output_[0]->data_cptr();
    auto target = input[1]->data_cptr();
    auto count = sigmoid_output_[0]->size();
    auto diff = input[0]->mutable_diff_cptr();

    vector_sub(count, sigmoid_output, target, diff);
    vector_scal(count, (T)1.0/input[0]->shape(0), diff);
}
}