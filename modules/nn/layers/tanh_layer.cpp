#include "tanh_layer.h"
#include <glog/logging.h>

namespace alchemy {
template<typename T>
void TanhLayer<T>::setup(const vector<Tensor<T> *> &input,
                         const vector<Tensor<T> *> &output)
{
    LOG(INFO) << "Setting up " << this->param_.name();
    LOG(INFO) << "input  #0: "  << input[0]->shape();

    output[0]->reshape(input[0]->shape());
    LOG(INFO) << "output #0: "  << output[0]->shape();
}

template<typename T>
void TanhLayer<T>::ForwardCPU(const vector<Tensor<T> *> &input,
                              const vector<Tensor<T> *> &output)
{
    auto input_data = input[0]->cpu_data();
    auto count = input[0]->count();
    auto output_data = output[0]->cpu_data();

    for(auto i = 0; i < count; ++i) {
        output_data[i] = std::tanh(input_data[i]);
    }
}

template<typename T>
void TanhLayer<T>::BackwardCPU(const vector<Tensor<T> *> &input,
                               const vector<Tensor<T> *> &output)
{
    auto count = input[0]->count();
    auto output_data = output[0]->cpu_data();
    auto input_diff = input[0]->cpu_diff();
    auto output_diff = output[0]->cpu_diff();

    for(auto i = 0; i < count; ++i) {
        auto tx = output_data[i];
        input_diff[i] = output_diff[i] *(1 - tx * tx);
    }
}

template class TanhLayer<float>;
template class TanhLayer<double>;

}