#ifndef ALCHEMY_NN_LAYERS_SIGMOID_LAYER_CUH
#define ALCHEMY_NN_LAYERS_SIGMOID_LAYER_CUH

#include <device_launch_parameters.h>

namespace alchemy {

template <typename T>
__global__ void sigmoid(const size_t size, const T* A, T* B)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        B[i] = 1.0/(1.0 + std::exp(-(A[i])));
    }
}

template <typename Device, typename T>
void SigmoidLayer<Device, T>::ForwardGPU(const vector<container *> &input,
                                 const vector<container *> &output)
{
    auto input_data = input[0]->data_gptr();
    auto count = input[0]->size();
    auto output_data = output[0]->mutable_data_gptr();

    sigmoid<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, input_data, output_data);
}


template <typename T>
__global__ void dsigmoid(const size_t size, const T* OutputData, const T* OutputDiff,  T* InputDiff)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        auto sv = OutputData[i];
        InputDiff[i] = OutputDiff[i] * sv * (1.0 - sv);
    }
}

template <typename Device, typename T>
void SigmoidLayer<Device, T>::BackwardGPU(const vector<container *> &input,
                                  const vector<container *> &output)
{
    auto count = input[0]->size();
    auto output_data = output[0]->data_gptr();
    auto output_diff = output[0]->diff_gptr();
    auto input_diff = input[0]->mutable_diff_gptr();

    dsigmoid<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, output_data, output_diff, input_diff);
}
}

#endif//! ALCHEMY_NN_LAYERS_SIGMOID_LAYER_CUH