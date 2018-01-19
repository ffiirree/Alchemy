#include <device_launch_parameters.h>
#include <math/math_op.h>
#include "accuracy_layer.h"

namespace alchemy {

template <typename T>
__global__ void max_index_kernel(const int count, const T * ptr, int * index){
    T max_value = 0;
    for(auto j = 0; j < count; ++j) {
        if(max_value < ptr[j]) {
            max_value = ptr[j];
            *index = j;
        };
    }
}

template<typename T>
void AccuracyLayer<T>::ForwardGPU(const vector<Tensor<T> *> &input,
                                  const vector<Tensor<T> *> &output)
{
    auto size = input[0]->shape(2) * input[0]->shape(3);
    auto o_ptr = input[0]->gpu_data();
    auto g_ptr = input[1]->gpu_data();
    int result_ = 0;
    Tensor<int> index_1({1}), index_2({1});

    for(auto i = 0; i < input[0]->shape(0); ++i) {

        max_index_kernel<<<1, 1>>>(size, o_ptr, index_1.gpu_data());
        max_index_kernel<<<1, 1>>>(size, g_ptr, index_2.gpu_data());

        auto _1 = index_1.cpu_data()[0];
        auto _2 = index_2.cpu_data()[0];
//        cudaDeviceSynchronize();
        if(_1 == _2)
            result_++;

        o_ptr += size;
        g_ptr += size;
    }

    /// cpu
    output[0]->cpu_data()[1] += result_;
    output[0]->cpu_data()[2] += input[0]->shape(0);
    output[0]->cpu_data()[0] = output[0]->cpu_data()[1] / output[0]->cpu_data()[2];
}

template void AccuracyLayer<float>::ForwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
template void AccuracyLayer<double>::ForwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
}