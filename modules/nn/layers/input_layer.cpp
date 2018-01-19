#include "input_layer.h"
#include <glog/logging.h>

namespace alchemy {

template<typename T>
void InputLayer<T>::setup(const vector<Tensor<T> *> &input,
                          const vector<Tensor<T> *> &output)
{
    LOG(INFO) << "Setting up " << this->param_.name();

    output[0]->reshape({ static_cast<int>(input_param_.batch_size()),
                         input_param_.data()[0].first.channels(),
                         input_param_.data()[0].first.rows,
                         input_param_.data()[0].first.cols
                       });
    LOG(INFO) << "output #0: "  << output[0]->shape();
    output[1]->reshape({ static_cast<int>(input_param_.batch_size()), 1, 10, 1 }); //TODO: label 暂时这样写着
    LOG(INFO) << "output #1: " << output[1]->shape();

    //
    data_ = input_param_.data();
    data_num_ = data_.size();
}

template<typename T>
void InputLayer<T>::ForwardCPU(const vector<Tensor<T>*>& input,
                               const vector<Tensor<T>*>& output)
{
    /// data
    auto data_ptr = output[0]->cpu_data();
    auto data_count = data_[0].first.size();
    auto data_size = data_count * sizeof(T);

    /// label
    auto label_ptr = output[1]->cpu_data();
    auto label_size = 10 * sizeof(T);

    for(size_t i = 0; i < input_param_.batch_size(); ++i, ++index_) {
        index_ %= data_num_;
        if(!index_) shuffle();

        auto& item = data_[index_];

        const auto& image = _Matrix<T>(item.first) * input_param_.scale();
        memmove(data_ptr, image.data, data_size);
        data_ptr += data_count;

        _Matrix<T> temp(10, 1, 1, (T)0);
        temp.at(item.second) = 1;
        memmove(label_ptr, temp.data, label_size);
        label_ptr += 10;
    }
}

template class InputLayer<float>;
template class InputLayer<double>;
}