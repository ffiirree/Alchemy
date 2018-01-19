#include "sigmoid_layer.h"
#include <device_launch_parameters.h>

namespace alchemy {

template <typename T>
__global__ void sigmoid(const size_t size, const T* A, T* B)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        B[i] = 1.0/(1.0 + std::exp(-(A[i])));
    }
}

template<typename T>
void SigmoidLayer<T>::ForwardGPU(const vector<Tensor<T> *> &input,
                                 const vector<Tensor<T> *> &output)
{
    auto input_data = input[0]->gpu_data();
    auto count = input[0]->count();
    auto output_data = output[0]->gpu_data();

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

template<typename T>
void SigmoidLayer<T>::BackwardGPU(const vector<Tensor<T> *> &input,
                                  const vector<Tensor<T> *> &output)
{
    const auto count = input[0]->count();
    const auto output_data = output[0]->gpu_data();
    auto input_diff = input[0]->gpu_diff();
    const auto output_diff = output[0]->gpu_diff();

    dsigmoid<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, output_data, output_diff, input_diff);
}

template void SigmoidLayer<float>::ForwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
template void SigmoidLayer<double>::ForwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
template void SigmoidLayer<float>::BackwardGPU(const vector<Tensor<float> *> &input, const vector<Tensor<float> *> &output);
template void SigmoidLayer<double>::BackwardGPU(const vector<Tensor<double> *> &input, const vector<Tensor<double> *> &output);
}