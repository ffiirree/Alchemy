#ifndef ALCHEMY_NN_OPTIMIZER_SGD_OPTIMIZER_H
#define ALCHEMY_NN_OPTIMIZER_SGD_OPTIMIZER_H

#include "nn/optimizer.h"

namespace alchemy {

template <typename Device, typename T>
class SgdOptimizer : public Optimizer<Device, T> {
public:
    explicit SgdOptimizer(const OptimizerParameter &param) : Optimizer<Device, T>(param) {}
    virtual ~SgdOptimizer() = default;

    virtual void update() override;
};

template <typename Device, typename T>
void SgdOptimizer<Device, T>::update()
{
    const auto& learnable_params = this->net_->learnable_params();
    for(auto& param : learnable_params) {
        axpy((T)-std::get<1>(param), std::get<0>(param)->diff(), std::get<0>(param)->data());
    }
}
}

#endif //! ALCHEMY_NN_OPTIMIZER_SGD_OPTIMIZER_H
