#include "sgd_optimizer.h"
#include "math/math_op.h"

namespace alchemy {

template<typename T>
void SgdOptimizer<T>::optimize()
{
    for(auto iter = 0; iter < this->param_.max_iter(); ++iter) {
        this->net_->Forward();
        this->net_->Backward();

        this->regularize();
        update();

        if(iter && iter % this->param_.test_interval() == 0) {

            for(auto test_iter = 0; test_iter < this->param_.test_iter(); ++test_iter) {
                this->test_net_->Forward();
            }
            LOG(INFO) << "Iteration " << std::setw(6) << std::setfill(' ') << iter << " : accuracy=" << this->test_net_->accuracy();
        }
    }
}

template<typename T>
void SgdOptimizer<T>::update()
{
    const auto& learnable_params = this->net_->learnable_params();
    if(Global::mode() == Global::CPU) {
        for(auto& param : learnable_params) {
            vector_axpy(std::get<0>(param)->count(), (T)-std::get<1>(param), std::get<0>(param)->cpu_diff(), std::get<0>(param)->cpu_data());
        }
    }
    else {
        for(auto& param : learnable_params) {
            vector_axpy_gpu(std::get<0>(param)->count(), (T)-std::get<1>(param), std::get<0>(param)->gpu_diff(), std::get<0>(param)->gpu_data());
        }
    }

}

template class SgdOptimizer<float>;
//template class SgdOptimizer<double>;
}
