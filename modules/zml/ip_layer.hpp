#ifndef _ZML_IP_LAYER_HPP
#define _ZML_IP_LAYER_HPP

extern "C" {
#include "cblas.h"
};
#include "layer.hpp"
#include "math_op.hpp"

namespace z {
template <typename T>
class InnerProductLayer: public Layer<T> {
    using container_type = Tensor<T>;
public:
    InnerProductLayer() = default;
    InnerProductLayer(int num, int chs, int rows, int cols);
    InnerProductLayer(const InnerProductLayer&)= delete;
    InnerProductLayer&operator=(const InnerProductLayer&)= delete;
    ~InnerProductLayer() = default;

    virtual void setup(const vector<container_type*>&input, const vector<container_type*>&output);

    virtual void ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);

private:
    Tensor<T> weights_;
    Tensor<T> biases_;
    Tensor<T> biasmer_;

    int M_;
    int N_;
    int K_;
};

template<typename T>
void InnerProductLayer<T>::setup(const vector<container_type *> &input, const vector<container_type *> &output)
{
    LOG(INFO) << "Inner Product Init: " << this->shape_[0] << " " << this->shape_[1] << " " << this->shape_[2] << " " << this->shape_[3];

    output[0]->reshape(this->shape_);
    biasmer_.reshape({input[0]->shape(0)});
    weights_.reshape({ this->shape_[2], input[0]->shape(2) });
    biases_.reshape({ this->shape_[2] });

    vector_set(input[0]->shape(0), 1.0, biasmer_.data());

    std::default_random_engine random_engine(time(nullptr));
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
    N_ = this->shape_[2];
    K_ = input[0]->shape(2);
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
    vector_axpy(weights_.count(), (T)-2./this->shape_[0], weights_.diff(), weights_.data());
    vector_axpy(biases_.count(), (T)-2./this->shape_[0], biases_.diff(), biases_.data());
}

template<typename T>
InnerProductLayer<T>::InnerProductLayer(int num, int chs, int rows, int cols)
        :Layer<T>()
{
    this->shape_.resize(4);
    this->shape_.at(0) = num;
    this->shape_.at(1) = chs;
    this->shape_.at(2) = rows;
    this->shape_.at(3) = cols;
}
}


#endif //_ZML_IP_LAYER_HPP
