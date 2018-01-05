#include <glog/logging.h>
#include <random>
#include <iostream>
#include "zml/util/math_op.hpp"
#include "ip_layer.hpp"

namespace z {

template<typename T>
void InnerProductLayer<T>::setup(const vector<container_type *> &input, const vector<container_type *> &output)
{
    output[0]->reshape({ input[0]->shape(0), input[0]->shape(1), static_cast<int>(param_.ip_param().neuron_size()), 1 });
    biasmer_.reshape({ input[0]->shape(0) });
    weights_.reshape({ param_.ip_param().neuron_size(), input[0]->shape(2) });
    biases_.reshape({ param_.ip_param().neuron_size() });

    vector_set(input[0]->shape(0), (T)1.0, biasmer_.data());

    std::default_random_engine random_engine(static_cast<unsigned long>(time(nullptr)));
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    auto weight_data = weights_.data();
    for(auto i = 0; i < weights_.count(); ++i) {
        weight_data[i] = distribution(random_engine);
    }
    auto bias_data = biases_.data();
    for(auto i = 0; i < biases_.count(); ++i) {
        bias_data[i] = distribution(random_engine);
    }

    /// N x C x R x C
    M_ = input[0]->shape(0);
    N_ = static_cast<int>(param_.ip_param().neuron_size());
    K_ = input[0]->shape(2);

    LOG(INFO) << "Inner Product Init: " << output[0]->shape(0) << " " << output[0]->shape(1) << " " << output[0]->shape(2) << " " << output[0]->shape(3);
}

template<typename T>
void InnerProductLayer<T>::ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output)
{
    auto input_data = input[0]->data();
    auto weight = this->weights_.data();
    auto count = input[0]->count();
    auto output_data = output[0]->data();

    // w * x
    matrix_mul(CblasNoTrans, CblasTrans,
               M_, N_, K_,
               (T)1, input_data, weight,
               (T)0, output_data);

    // output_data += bias
    matrix_mul(CblasNoTrans, CblasNoTrans,
               M_, N_, 1,
               (T)1, biasmer_.data(), biases_.data(),
               (T)1, output_data);
}

template<typename T>
void InnerProductLayer<T>::BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output)
{
    // 向前传递误差
    matrix_mul(CblasNoTrans, CblasNoTrans,
               M_, K_, N_,
               (T)1, output[0]->diff(), weights_.data(),
               (T)0, input[0]->diff());


    // 计算参数的更新值
    // weights
    matrix_mul(CblasTrans, CblasNoTrans,
               N_, K_, M_,
               (T)1, output[0]->diff(), input[0]->data(),
               (T)0, weights_.diff());

    // biases
    matvec_mul(CblasTrans, M_, N_,
               (T)1, output[0]->diff(), biasmer_.data(),
               (T)0, biases_.diff());

    // update weights & biases
    vector_axpy(weights_.count(), (T)-inner_product_param_.wlr()/input[0]->shape(0), weights_.diff(), weights_.data());
    vector_axpy(biases_.count(), (T)-inner_product_param_.blr()/input[0]->shape(0), biases_.diff(), biases_.data());
}

template class InnerProductLayer<float>;
template class InnerProductLayer<double>;
}