#include <zml/util/math_op.hpp>
#include <glog/logging.h>
#include "softmax_layer.hpp"

namespace z {

template<typename T>
void SoftmaxLayer<T>::setup(const vector<container_type *> &input, const vector<container_type *> &output)
{
    LOG(INFO) << "Setting up " << param_.name();
    LOG(INFO) << "input  #0: "  << input[0]->shape();

    output[0]->reshape(input[0]->shape());
    LOG(INFO) << "output #0: "  << output[0]->shape();

    sum_.reshape(input[0]->shape());
    sum_multer_.reshape({ input[0]->shape(2), input[0]->shape(2) });

    vector_set(sum_multer_.count(), (T)1., sum_multer_.data());
}

template<typename T>
void SoftmaxLayer<T>::ForwardCPU(const vector<container_type *> &input,
                                 const vector<container_type *> &output)
{
    const auto count = input[0]->count();
    auto input_data = input[0]->data();
    auto output_data = output[0]->data();

    vector_copy(count, input_data, output_data);

    //TODO: numerical issues
    // exp
    vector_exp(count, output_data, output_data);
    // sum
    matrix_mul(CblasNoTrans, CblasNoTrans,
               input[0]->shape(0), input[0]->shape(2), input[0]->shape(2),
               (T)1., output_data, sum_multer_.data(),
               (T)0., sum_.data());
    // div
    vector_div(count, output_data, sum_.data(), output_data);
}

template<typename T>
void SoftmaxLayer<T>::BackwardCPU(const vector<container_type *> &input,
                                  const vector<container_type *> &output)
{
    LOG(FATAL) << "Not implement!";
}


template class SoftmaxLayer<float>;
template class SoftmaxLayer<double>;
}