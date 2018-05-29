#ifndef ALCHEMY_NN_OPTIMIZER_SGD_OPTIMIZER_H
#define ALCHEMY_NN_OPTIMIZER_SGD_OPTIMIZER_H

#include "nn/optimizer.h"

namespace alchemy {

template <typename Device, typename T>
class SgdOptimizer : public Optimizer<Device, T> {
public:
    explicit SgdOptimizer(const OptimizerParameter &param) : Optimizer<Device, T>(param) {}
    virtual ~SgdOptimizer() = default;

    virtual void optimize() override;
    virtual void update() override;
};

template <typename Device, typename T>
void SgdOptimizer<Device, T>::optimize()
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
            LOG(INFO) << "Iteration " << std::setw(6) << std::setfill(' ') << iter
                      << " : accuracy=" << std::setw(9) << std::left << std::setfill(' ') << this->test_net_->accuracy()
                      << " , loss=" << this->net_->loss();
        }
    }
}

template <typename Device, typename T>
void SgdOptimizer<Device, T>::update()
{
    const auto& learnable_params = this->net_->learnable_params();
    if(Global::mode() == Global::CPU) {
        for(auto& param : learnable_params) {
            vector_axpy(std::get<0>(param)->size(), (T)-std::get<1>(param), std::get<0>(param)->diff_cptr(), std::get<0>(param)->mutable_data_cptr());
        }
    }
    else {
        for(auto& param : learnable_params) {
            vector_axpy_gpu(std::get<0>(param)->size(), (T)-std::get<1>(param), std::get<0>(param)->diff_gptr(), std::get<0>(param)->mutable_data_gptr());
        }
    }
}
}

#endif //! ALCHEMY_NN_OPTIMIZER_SGD_OPTIMIZER_H
