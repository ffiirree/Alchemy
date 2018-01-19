#include "sigmoid_cross_entropy_loss_layer.h"
#include <glog/logging.h>
#include "math/math_op.h"
#include "sigmoid_layer.h"

namespace alchemy {

template<typename T>
void SigmoidCrossEntropyLossLayer<T>::setup(const vector<Tensor<T> *> &input,
                                            const vector<Tensor<T> *> &output)
{
    LOG(INFO) << "Setting up " << this->param_.name();
    LOG(INFO) << "input  #0: "  << input[0]->shape();
    LOG(INFO) << "input  #1: "  << input[1]->shape();
    LOG_IF(FATAL, input.size() < 2) << "input size: " << input.size();

    sigmoid_layers_ = shared_ptr<Layer<T>>(
            new SigmoidLayer<T>(
                    LayerParameter()
                            .type(SIGMOID_LAYER)
                            .sigmoid_param(
                                    SigmoidParameter()
                            )
            ));
    sigmoid_output_.push_back(shared_ptr<Tensor<T>>(new Tensor<T>()));
    sigmoid_layers_->setup(input, { sigmoid_output_[0].get() });

    output[0]->reshape({ 1 });
    LOG(INFO) << "output #0: "  << output[0]->shape();
}


template<typename T>
void SigmoidCrossEntropyLossLayer<T>::ForwardCPU(const vector<Tensor<T> *> &input,
                                                 const vector<Tensor<T> *> &output)
{
    // computes the sigmoid outputs.
    sigmoid_layers_->Forward(input, { sigmoid_output_[0].get() });

    //TODO: loss
}

template<typename T>
void  SigmoidCrossEntropyLossLayer<T>::BackwardCPU(const vector<Tensor<T> *> &input,
                                                   const vector<Tensor<T> *> &output)
{
    auto sigmoid_output = sigmoid_output_[0]->cpu_data();
    auto target = input[1]->cpu_data();
    auto count = sigmoid_output_[0]->count();
    auto diff = input[0]->cpu_diff();

    vector_sub(count, sigmoid_output, target, diff);
    vector_scal(count, (T)1.0/input[0]->shape(0), diff);
}

template class SigmoidCrossEntropyLossLayer<float>;
template class SigmoidCrossEntropyLossLayer<double>;

}