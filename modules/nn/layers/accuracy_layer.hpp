#include <algorithm>
#include <math/math_op.h>

namespace alchemy {

template <typename Device, typename T>
void AccuracyLayer<Device, T>::setup(const vector<container *> &input,
                             const vector<container *> &output)
{
    output[0]->reshape({ 3 });
    vector_set(output[0]->size(), (T)0., output[0]->mutable_data_cptr());
}

template <typename Device, typename T>
void AccuracyLayer<Device, T>::ForwardCPU(const vector<container *> &input,
                                  const vector<container *> &output)
{
    auto size = input[0]->size(2, 4);
    auto o_ptr = input[0]->data_cptr();
    auto g_ptr = input[1]->data_cptr();
    int result_ = 0;
    for(size_t i = 0; i < input[0]->shape(0); ++i) {
        // simples
        auto o_iter = std::max_element(o_ptr + i * size, o_ptr + i * size + size);
        auto g_iter = std::max_element(g_ptr + i * size, g_ptr + i * size + size);
        if(std::distance(o_ptr + i * size, o_iter) == std::distance(g_ptr + i * size, g_iter)) {
            result_++;
        }
    }

    output[0]->mutable_data_cptr()[1] += result_;
    output[0]->mutable_data_cptr()[2] += input[0]->shape(0);
    output[0]->mutable_data_cptr()[0] = output[0]->data_cptr()[1] / output[0]->data_cptr()[2];
}
}
