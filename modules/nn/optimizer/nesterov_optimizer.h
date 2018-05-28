#ifndef ALCHEMY_NN_OPTIMIZER_NESTEROV_OPTIMIZER_H
#define ALCHEMY_NN_OPTIMIZER_NESTEROV_OPTIMIZER_H

#include "nn/optimizer.h"

namespace alchemy {

template <typename Device, typename T>
class NesterovOptimizer : public Optimizer<Device, T> {
public:
    explicit NesterovOptimizer(const OptimizerParameter &param);
    virtual ~NesterovOptimizer() = default;

    virtual void optimize();
    virtual void update();

protected:
    vector<Tensor<T>> buf_;
    vector<Tensor<T>> buf2_;
};

template <typename Device, typename T>
NesterovOptimizer<Device, T>::NesterovOptimizer(const OptimizerParameter &param)
        : Optimizer<Device, T>(param)
{
    const auto& learnable_params = this->net_->learnable_params();
    for(const auto& learnable_param : learnable_params) {
        const auto& shape = std::get<0>(learnable_param)->shape();
        Tensor<T> buf(shape), buf2(shape);
        Filler<T>::constant_fill(buf.count(), buf.mutable_cptr(), 0.0);
        buf_.push_back(buf);
        buf2_.push_back(buf2);
    }
}

template <typename Device, typename T>
void NesterovOptimizer<Device, T>::optimize()
{
    for(auto iter = 0; iter < this->param_.max_iter(); ++iter) {
        this->net_->Forward();
        this->net_->Backward();

        update();
        this->regularize();

        if(iter && iter % this->param_.test_interval() == 0) {

            for(auto test_iter = 0; test_iter < this->param_.test_iter(); ++test_iter) {
                this->test_net_->Forward();
            }
            LOG(INFO) << "Iteration " << std::setw(6) << std::setfill(' ') << iter << " : accuracy=" << this->test_net_->accuracy();
        }
    }
}

template <typename Device, typename T>
void NesterovOptimizer<Device, T>::update()
{
    const auto& learnable_params = this->net_->learnable_params();
    auto momentum = this->param_.momentum();

    if(Global::mode() == Global::CPU) {
        for(size_t idx = 0; idx < learnable_params.size(); ++idx) {
            // v_ = v
            vector_copy(buf2_[idx].count(), buf_[idx].cptr(), buf2_[idx].mutable_cptr());
            // v_ = m * v_
            vector_scal(buf2_[idx].count(), (T)momentum, buf2_[idx].mutable_cptr());
            // v_ = v_ -
            vector_axpy(buf2_[idx].count(), (T)-std::get<1>(learnable_params[idx]), std::get<0>(learnable_params[idx])->diff_cptr(), buf2_[idx].mutable_cptr());
            // v = v_
            vector_copy(buf2_[idx].count(), buf2_[idx].cptr(), buf_[idx].mutable_cptr());
            vector_axpy(buf2_[idx].count(), (T)(1.0 + momentum), buf_[idx].cptr(), std::get<0>(learnable_params[idx])->mutable_data_cptr());
        }
    }
    else {
        for(size_t idx = 0; idx < learnable_params.size(); ++idx) {
            vector_copy_gpu(buf2_[idx].count(), buf_[idx].gptr(), buf2_[idx].mutable_gptr());
            vector_scal_gpu(buf2_[idx].count(), (T)momentum, buf2_[idx].mutable_gptr());
            vector_axpy_gpu(buf2_[idx].count(), (T)-std::get<1>(learnable_params[idx]), std::get<0>(learnable_params[idx])->diff_gptr(), buf2_[idx].mutable_gptr());
            vector_copy_gpu(buf2_[idx].count(), buf2_[idx].gptr(), buf_[idx].mutable_gptr());
            vector_axpy_gpu(buf2_[idx].count(), (T)1.0, buf_[idx].gptr(), std::get<0>(learnable_params[idx])->mutable_data_gptr());
        }
    }
}
}
#endif //!ALCHEMY_NN_OPTIMIZER_NESTEROV_OPTIMIZER_H
