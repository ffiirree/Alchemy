#include <device_launch_parameters.h>
#include "relu_layer.hpp"

namespace z {

template <typename T>
__global__ void relu_kernel(const size_t size, const T* InputData, double alpha, T* OutputData)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        OutputData[i] = InputData[i] > (T)0.0 ? InputData[i] : alpha * InputData[i];
    }
}

template<typename T>
void ReLuLayer<T>::ForwardGPU(const vector<container_type *> &input,
                              const vector<container_type *> &output)
{
    auto count = input[0]->count();
    auto input_data = input[0]->gpu_data();
    auto output_data = output[0]->gpu_data();

    relu_kernel<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, input_data, relu_param_.alpha(), output_data);
}


template <typename T>
__global__ void drelu_kernel(const size_t size, const T* InputData, const T* OutputDiff, double alpha, T* InputDiff)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        InputDiff[i] = OutputDiff[i] * ((InputData[i] > 0) ? 1 : alpha);
    }
}

template<typename T>
void ReLuLayer<T>::BackwardGPU(const vector<container_type *> &input,
                               const vector<container_type *> &output)
{
    auto count = input[0]->count();
    auto input_data = input[0]->gpu_data();
    auto input_diff = input[0]->gpu_diff();
    auto output_diff = output[0]->gpu_diff();

    drelu_kernel<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, input_data, output_diff, relu_param_.alpha(), input_diff);
}

template void ReLuLayer<float>::ForwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
template void ReLuLayer<double>::ForwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
template void ReLuLayer<float>::BackwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
template void ReLuLayer<double>::BackwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
}