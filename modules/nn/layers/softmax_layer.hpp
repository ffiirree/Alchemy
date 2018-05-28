#include "math/math_op.h"

namespace alchemy {

template<typename T>
void SoftmaxLayer<T>::setup(const vector<Blob<T> *> &input, const vector<Blob<T> *> &output)
{
    output[0]->reshape(input[0]->shape());

    sum_.reshape(input[0]->shape());
    sum_multer_.reshape({ input[0]->shape(2), input[0]->shape(2) });

    vector_set(sum_multer_.count(), (T)1., sum_multer_.mutable_data_cptr());
}

template<typename T>
void SoftmaxLayer<T>::ForwardCPU(const vector<Blob<T> *> &input,
                                 const vector<Blob<T> *> &output)
{
    const auto count = input[0]->count();
    auto input_data = input[0]->data_cptr();
    auto output_data = output[0]->mutable_data_cptr();

    vector_copy(count, input_data, output_data);

    //TODO: numerical issues
    // exp
    vector_exp(count, output_data, output_data);
    // sum
    matrix_mul(CblasNoTrans, CblasNoTrans,
               input[0]->shape(0), input[0]->shape(2), input[0]->shape(2),
               (T)1., output_data, sum_multer_.data_cptr(),
               (T)0., sum_.mutable_data_cptr());
    // div
    vector_div(count, output_data, sum_.data_cptr(), output_data);
}

template<typename T>
void SoftmaxLayer<T>::BackwardCPU(const vector<Blob<T> *> &input,
                                  const vector<Blob<T> *> &output)
{
    LOG(FATAL) << "Not implement!";
}
}