#include "tanh_layer.h"
#include <device_launch_parameters.h>

namespace alchemy {

template <typename T>
__global__ void tanh_kernel(const size_t size, const T* A, T* B)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        B[i] = std::tanh(A[i]);
    }
}
template<typename T>
void TanhLayer<T>::ForwardGPU(const vector<Tensor<T> *> &input,
                              const vector<Tensor<T> *> &output)
{
    const auto count = input[0]->count();
    const auto input_data = input[0]->gpu_data();
    auto output_data = output[0]->gpu_data();

    tanh_kernel<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, input_data, output_data);
}

template <typename T>
__global__ void dtanh_kernel(const size_t size, const T* OutputData, const T* OutputDiff, T* InputDiff)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        auto tx = OutputData[i];
        InputDiff[i] = OutputDiff[i] *(1 - tx * tx);
    }
}
template<typename T>
void TanhLayer<T>::BackwardGPU(const vector<Tensor<T> *> &input,
                               const vector<Tensor<T> *> &output)
{
    auto count = input[0]->count();
    auto output_data = output[0]->gpu_data();
    auto input_diff = input[0]->gpu_diff();
    auto output_diff = output[0]->gpu_diff();

    dtanh_kernel<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, output_data, output_diff, input_diff);
}

template void TanhLayer<float>::ForwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
template void TanhLayer<double>::ForwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
template void TanhLayer<float>::BackwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
template void TanhLayer<double>::BackwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
}