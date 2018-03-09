#include "sigmoid_layer.h"
#include <glog/logging.h>

namespace alchemy {

template<typename T>
void SigmoidLayer<T>::setup(const vector<Blob<T> *> &input,
                            const vector<Blob<T> *> &output)
{
    LOG(INFO) << "Setting up " << this->param_.name();
    LOG(INFO) << "input  #0: "  << input[0]->shape();

    output[0]->reshape(input[0]->shape());
    LOG(INFO) << "output #0: "  << output[0]->shape();
}

template <typename T>
inline T sigmoid(T value)
{
    return 1.0/(1.0 + std::exp(value));
}

template<typename T>
void SigmoidLayer<T>::ForwardCPU(const vector<Blob<T>*>& input,
                                 const vector<Blob<T>*>& output)
{
    auto input_data = input[0]->data_cptr();
    auto count = input[0]->count();
    auto output_data = output[0]->data_cptr();

    for(auto i = 0; i < count; ++i) {
        output_data[i] = sigmoid(-1. * input_data[i]);
    }
}

template<typename T>
void SigmoidLayer<T>::BackwardCPU(const vector<Blob<T>*>& input,
                                  const vector<Blob<T>*>& output)
{
    auto count = input[0]->count();
    auto output_data = output[0]->data_cptr();
    auto input_diff = input[0]->diff_cptr();
    auto output_diff = output[0]->diff_cptr();

    for(auto i = 0; i < count; ++i) {
        auto sv = output_data[i];
        input_diff[i] = output_diff[i] * sv * (1.0 - sv);
    }
}

template class SigmoidLayer<float>;
template class SigmoidLayer<double>;
}