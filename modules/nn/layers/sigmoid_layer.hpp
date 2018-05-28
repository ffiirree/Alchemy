#ifndef ALCHEMY_NN_LAYERS_SIGMOID_LAYER_HPP
#define ALCHEMY_NN_LAYERS_SIGMOID_LAYER_HPP

namespace alchemy {

template<typename T>
void SigmoidLayer<T>::setup(const vector<container *> &input,
                            const vector<container *> &output)
{
    output[0]->reshape(input[0]->shape());
}

template <typename T>
inline T sigmoid(T value)
{
    return 1.0/(1.0 + std::exp(-value));
}

template<typename T>
void SigmoidLayer<T>::ForwardCPU(const vector<container *>& input,
                                 const vector<container *>& output)
{
    auto input_data = input[0]->data_cptr();
    auto count = input[0]->count();
    auto output_data = output[0]->mutable_data_cptr();

    for(auto i = 0; i < count; ++i) {
        output_data[i] = sigmoid(input_data[i]);
    }
}

template<typename T>
void SigmoidLayer<T>::BackwardCPU(const vector<container *>& input,
                                  const vector<container *>& output)
{
    auto count = input[0]->count();
    auto output_data = output[0]->data_cptr();
    auto input_diff = input[0]->mutable_diff_cptr();
    auto output_diff = output[0]->diff_cptr();

    for(auto i = 0; i < count; ++i) {
        auto sv = output_data[i];
        input_diff[i] = output_diff[i] * sv * (1.0 - sv);
    }
}
}

#endif//! ALCHEMY_NN_LAYERS_SIGMOID_LAYER_HPP