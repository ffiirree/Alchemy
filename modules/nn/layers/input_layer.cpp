#include "input_layer.h"
#include <glog/logging.h>
#include "math/math_op.h"

namespace alchemy {

template<typename T>
void InputLayer<T>::setup(const vector<Blob<T> *> &input,
                          const vector<Blob<T> *> &output)
{
    LOG(INFO) << "Setting up: " << this->param_.name();

    output[0]->reshape({ static_cast<int>(input_param_.batch_size()),
                         static_cast<int>(data_.chs()),
                         static_cast<int>(data_.rows()),
                         static_cast<int>(data_.cols())
                       });
    LOG(INFO) << "output #0: "  << output[0]->shape();
    output[1]->reshape({ static_cast<int>(input_param_.batch_size()),
                         1,
                         static_cast<int>(data_.classification_num()),
                         1
                       });
    LOG(INFO) << "output #1: " << output[1]->shape();

    data_num_ = data_.size();
    vector_scal(static_cast<const int>(data_.size() * data_.image_size()),
                (T)input_param_.scale(),
                data_.images().get());
}

template<typename T>
void InputLayer<T>::ForwardCPU(const vector<Blob<T>*>& input,
                               const vector<Blob<T>*>& output)
{
    auto batch_size = input_param_.batch_size();
    /// data
    auto images_ptr = data_.images().get();
    memmove(output[0]->mutable_data_cptr(),
            images_ptr + index_ * data_.image_size(),
            batch_size * data_.image_size() * sizeof(T));

    /// label
    auto labels_ptr = data_.labels().get();
    memmove(output[1]->mutable_data_cptr(),
            labels_ptr + index_ * data_.label_size(),
            batch_size * data_.label_size() * sizeof(T));

    index_ = (index_ + batch_size) % data_num_;
    if(data_num_ - index_ < batch_size) index_ = 0;
}

template class InputLayer<float>;
template class InputLayer<double>;
}