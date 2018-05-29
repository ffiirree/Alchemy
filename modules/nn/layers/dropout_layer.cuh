#include <device_launch_parameters.h>

namespace alchemy {

template <typename T>
__global__ void mul_kernel(int count, const T* A, const T* B, T* C)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        C[i] = A[i] * B[i];
    }
}

template <typename Device, typename T>
void DropoutLayer<Device, T>::ForwardGPU(const vector<container *> &input,
                                 const vector<container *> &output)
{
    const auto count = input[0]->size();
    const auto input_data = input[0]->data_gptr();
    auto output_data = output[0]->mutable_data_gptr();

    if(this->param_.phase() == TRAIN) {
        Filler<Device, T>::bernoulli_fill(filter_.size(), filter_.mutable_cptr(), 0.5);
        const auto filter_data = filter_.gptr();
        
        mul_kernel<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, input_data, filter_data, output_data);
    }
    else{
        vector_copy_gpu(count, input_data, output_data);
    }
}

template <typename Device, typename T>
void DropoutLayer<Device, T>::BackwardGPU(const vector<container *> &input,
                                  const vector<container *> &output)
{
    auto count = input[0]->size();
    auto input_diff = input[0]->mutable_diff_gptr();
    auto output_diff = output[0]->diff_gptr();

    if(this->param_.phase() == TRAIN) {
        const auto filter_data = filter_.gptr();

        mul_kernel<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, output_diff, filter_data, input_diff);
    }
    else {
        vector_copy_gpu(count, output_diff, input_diff);
    }
}
}