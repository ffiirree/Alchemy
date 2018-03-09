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
void SigmoidLayer<T>::ForwardGPU(const vector<Blob<T> *> &input,
                                 const vector<Blob<T> *> &output)
{
    auto input_data = input[0]->data_gptr();
    auto count = input[0]->count();
    auto output_data = output[0]->data_gptr();

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
void SigmoidLayer<T>::BackwardGPU(const vector<Blob<T> *> &input,
                                  const vector<Blob<T> *> &output)
{
    const auto count = input[0]->count();
    const auto output_data = output[0]->data_gptr();
    auto input_diff = input[0]->diff_gptr();
    const auto output_diff = output[0]->diff_gptr();

    dsigmoid<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, output_data, output_diff, input_diff);
}

template void SigmoidLayer<float>::ForwardGPU(const vector<Blob<float> *> &input, const vector<Blob<float> *> &output);
template void SigmoidLayer<double>::ForwardGPU(const vector<Blob<double> *> &input, const vector<Blob<double> *> &output);
template void SigmoidLayer<float>::BackwardGPU(const vector<Blob<float> *> &input, const vector<Blob<float> *> &output);
template void SigmoidLayer<double>::BackwardGPU(const vector<Blob<double> *> &input, const vector<Blob<double> *> &output);
}