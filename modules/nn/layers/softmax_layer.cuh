#include "math/math_op.h"

namespace alchemy {

template <typename Device, typename T>
void SoftmaxLayer<Device, T>::ForwardGPU(const vector<container *> &input,
                                 const vector<container *> &output)
{
    const auto count = input[0]->size();
    auto input_data = input[0]->data_gptr();
    auto output_data = output[0]->mutable_data_gptr();

    vector_copy_gpu(count, input_data, output_data);

    //TODO: numerical issues
    // exp
    vector_exp_gpu(count, output_data, output_data);
    // sum
    matrix_mul_gpu(CblasNoTrans, CblasNoTrans,
                   input[0]->shape(0), input[0]->shape(2), input[0]->shape(2),
                   (T)1., output_data, sum_multer_.data_gptr(),
                   (T)0., sum_.mutable_data_gptr());
    // div
    vector_div_gpu(count, output_data, sum_.data_gptr(), output_data);
}

template <typename Device, typename T>
void SoftmaxLayer<Device, T>::BackwardGPU(const vector<container *> &input,
                                  const vector<container *> &output)
{
    LOG(FATAL) << "Not implement!";
}
}