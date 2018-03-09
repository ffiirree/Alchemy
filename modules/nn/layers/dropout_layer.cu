#include "dropout_layer.h"
#include <device_launch_parameters.h>

namespace alchemy {

template <typename T>
__global__ void mul_kernel(int count, const T* A, const T* B, T* C)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        C[i] = A[i] * B[i];
    }
}

template<typename T>
void DropoutLayer<T>::ForwardGPU(const vector<Blob<T> *> &input,
                                 const vector<Blob<T> *> &output)
{
    const auto count = input[0]->count();
    const auto input_data = input[0]->data_gptr();
    auto output_data = output[0]->data_gptr();

    if(this->param_.phase() == TRAIN) {
        Filler<T>::bernoulli_fill(filter_.count(), filter_.cptr(), 0.5);
        const auto filter_data = filter_.gptr();
        
        mul_kernel<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, input_data, filter_data, output_data);
    }
    else{
        vector_copy_gpu(count, input_data, output_data);
    }
}

template<typename T>
void DropoutLayer<T>::BackwardGPU(const vector<Blob<T> *> &input,
                                  const vector<Blob<T> *> &output)
{
    const auto count = input[0]->count();
    auto input_diff = input[0]->diff_gptr();
    const auto output_diff = output[0]->diff_gptr();

    if(this->param_.phase() == TRAIN) {
        const auto filter_data = filter_.gptr();

        mul_kernel<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, output_diff, filter_data, input_diff);
    }
    else {
        vector_copy_gpu(count, output_diff, input_diff);
    }
}

template void DropoutLayer<float>::ForwardGPU(const vector<Blob<float> *> &input, const vector<Blob<float> *> &output);
template void DropoutLayer<double>::ForwardGPU(const vector<Blob<double> *> &input, const vector<Blob<double> *> &output);
template void DropoutLayer<float>::BackwardGPU(const vector<Blob<float> *> &input, const vector<Blob<float> *> &output);
template void DropoutLayer<double>::BackwardGPU(const vector<Blob<double> *> &input, const vector<Blob<double> *> &output);
}