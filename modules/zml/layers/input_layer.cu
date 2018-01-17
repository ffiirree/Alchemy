#include "input_layer.hpp"

namespace z {

template<typename T>
void InputLayer<T>::ForwardGPU(const vector<container_type *> &input,
                                const vector<container_type *> &output)
{
    /// data
    auto data_ptr = output[0]->gpu_data();
    auto data_count = data_[0].first.size();
    auto data_size = data_count * sizeof(T);

    /// label
    auto label_ptr = output[1]->gpu_data();
    auto label_size = 10 * sizeof(T);

    for(size_t i = 0; i < input_param_.batch_size(); ++i, ++index_) {
        index_ %= data_num_;
        if(!index_) shuffle();

        auto& item = data_[index_];

        const auto& image = _Matrix<T>(item.first) * input_param_.scale();
        cudaMemcpy(data_ptr, image.data, data_size, cudaMemcpyHostToDevice);
//        memmove(data_ptr, image.data, data_size);
        data_ptr += data_count;

        _Matrix<T> temp(10, 1, 1, (T)0);
        temp.at(item.second) = 1;
        cudaMemcpy(label_ptr, temp.data, label_size, cudaMemcpyHostToDevice);
//        memmove(label_ptr, temp.data, label_size);
        label_ptr += 10;
    }
}

template void InputLayer<float>::ForwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
template void InputLayer<double>::ForwardGPU(const vector<container_type *> &input, const vector<container_type *> &output);
}