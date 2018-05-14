#include "input_layer.h"
#include <glog/logging.h>
#include "math/math_op.h"

namespace alchemy {

template<typename T>
void InputLayer<T>::setup(const vector<Blob<T> *> &input,
                          const vector<Blob<T> *> &output)
{
    LOG(INFO) << "Setting up: " << this->param_.name();

    auto source = input_param_.source();

    output[0]->reshape({ static_cast<int>(input_param_.batch_size()),
                         static_cast<int>(source->chs()),
                         static_cast<int>(source->rows()),
                         static_cast<int>(source->cols())
                       });
    LOG(INFO) << "output #0: "  << output[0]->shape();
    output[1]->reshape({ static_cast<int>(input_param_.batch_size()),
                         1,
                         static_cast<int>(source->classification_num()),
                         1
                       });
    LOG(INFO) << "output #1: " << output[1]->shape();

    data_num_ = source->size();
    vector_scal(static_cast<const int>(source->size() * source->image_size()),
                (T)input_param_.scale(),
                (T*)source->images().get());
}

template<typename T>
void InputLayer<T>::ForwardCPU(const vector<Blob<T>*>& input,
                               const vector<Blob<T>*>& output)
{
    auto batch_size = input_param_.batch_size();

    auto source = input_param_.source();
    if(!source->hasNext(static_cast<int>(batch_size))) source->reset();

    auto data_pair = source->next(static_cast<int>(batch_size));

    /// data
    memmove(output[0]->mutable_data_cptr(),
            data_pair.first,
            batch_size * source->image_size() * sizeof(T));

//    print_cpu(output[0]->count(), output[0]->data_cptr());

    /// label
    memmove(output[1]->mutable_data_cptr(),
            data_pair.second,
            batch_size * source->label_size() * sizeof(T));

//    print_cpu(output[1]->count(), output[1]->data_cptr());
}

template class InputLayer<float>;
template class InputLayer<double>;
}