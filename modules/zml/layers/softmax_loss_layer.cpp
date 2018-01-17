#include <glog/logging.h>
#include <zml/util/math_op.hpp>
#include "softmax_loss_layer.hpp"
#include "softmax_layer.hpp"


namespace z {

template<typename T>
void SoftmaxLossLayer<T>::setup(const vector<container_type *> &input,
                                const vector<container_type *> &output)
{
    LOG(INFO) << "Setting up " << param_.name();
    LOG(INFO) << "input  #0: "  << input[0]->shape();
    LOG_IF(FATAL, input.size() < 2) << "input size: " << input.size();

    softmax_layer_ = shared_ptr<Layer<T>>(
            new SoftmaxLayer<T>(
                    LayerParameter()
                            .type(SOFTMAX_LAYER)
                            .softmax_param(
                                    SoftmaxParameter()
                            )
            ));

    softmax_output_.push_back(shared_ptr<Tensor<T>>(new Tensor<T>()));
    softmax_layer_->setup(input, { softmax_output_[0].get() });

    output[0]->reshape({ 1 });
    LOG(INFO) << "output #0: "  << output[0]->shape();
}

template<typename T>
void SoftmaxLossLayer<T>::ForwardCPU(const vector<container_type *> &input,
                                     const vector<container_type *> &output)
{
    softmax_layer_->Forward(input, { softmax_output_[0].get() });

    //TODO: loss
}

template<typename T>
void SoftmaxLossLayer<T>::BackwardCPU(const vector<container_type *> &input,
                                      const vector<container_type *> &output)
{
    const auto count = input[0]->count();
    const auto label_data = input[1]->cpu_data();
    const auto input_data = input[0]->cpu_data();
    auto input_diff = input[0]->cpu_diff();

    vector_sub(count, input_data, label_data, input_diff);
    vector_scal(count, (T)1.0/input[0]->shape(0), input_diff);
}

template class SoftmaxLossLayer<float>;
template class SoftmaxLossLayer<double>;
}