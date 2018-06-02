#ifndef ALCHEMY_NN_OPTIMIZER_NESTEROV_OPTIMIZER_H
#define ALCHEMY_NN_OPTIMIZER_NESTEROV_OPTIMIZER_H

#include "nn/optimizer.h"

namespace alchemy {

template <typename Device, typename T>
class NesterovOptimizer : public Optimizer<Device, T> {
public:
    explicit NesterovOptimizer(const OptimizerParameter &param);
    virtual ~NesterovOptimizer() = default;

    virtual void update();

protected:
    vector<Tensor<Device, T>> buf_;
    vector<Tensor<Device, T>> buf2_;
};

template <typename Device, typename T>
NesterovOptimizer<Device, T>::NesterovOptimizer(const OptimizerParameter &param)
        : Optimizer<Device, T>(param)
{
    const auto& learnable_params = this->net_->learnable_params();
    for(const auto& learnable_param : learnable_params) {
        const auto& shape = std::get<0>(learnable_param)->shape();
        Tensor<Device, T> buf(shape), buf2(shape);
        Filler<Device, T>::constant_fill(buf.size(), buf.mutable_cptr(), 0.0);
        buf_.push_back(buf);
        buf2_.push_back(buf2);
    }
}

template <typename Device, typename T>
void NesterovOptimizer<Device, T>::update()
{
    const auto& learnable_params = this->net_->learnable_params();
    auto momentum = this->param_.momentum();
    for(size_t idx = 0; idx < learnable_params.size(); ++idx) {
        // v_ = v
        Copy(buf_[idx], buf2_[idx]);
        // v_ = m * v_
        Scale((T)momentum, buf2_[idx]);
        // v_ = v_ -
        axpy( (T)-std::get<1>(learnable_params[idx]), std::get<0>(learnable_params[idx])->diff(), buf2_[idx]);
        // v = v_
        Copy(buf2_[idx], buf_[idx]);
        axpy((T)(1.0 + momentum), buf_[idx], std::get<0>(learnable_params[idx])->data());
    }
}
}
#endif //!ALCHEMY_NN_OPTIMIZER_NESTEROV_OPTIMIZER_H
