#include <math/math_op.h>
#include "input_layer.h"

namespace alchemy {

template<typename T>
void InputLayer<T>::ForwardGPU(const vector<Blob<T> *> &input,
                                const vector<Blob<T> *> &output)
{
    auto batch_size = input_param_.batch_size();
    /// data
    auto images_ptr = data_.images().get();
    cudaMemcpy(output[0]->data_gptr(),
               images_ptr + index_ * data_.image_size(),
               batch_size * data_.image_size() * sizeof(T),
               cudaMemcpyHostToDevice);

    /// label
    auto labels_ptr = data_.labels().get();
    cudaMemcpy(output[1]->data_gptr(),
               labels_ptr + index_ * data_.label_size(),
               batch_size * data_.label_size() * sizeof(T),
               cudaMemcpyHostToDevice);

    index_ = (index_ + batch_size) % data_num_;
    if(data_num_ - index_ < batch_size) index_ = 0;
}

template void InputLayer<float>::ForwardGPU(const vector<Blob<float> *> &input, const vector<Blob<float> *> &output);
template void InputLayer<double>::ForwardGPU(const vector<Blob<double> *> &input, const vector<Blob<double> *> &output);
}