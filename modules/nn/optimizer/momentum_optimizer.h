#ifndef ALCHEMY_NN_OPTIMIZER_SGD_MOMENTUM_OPTIMIZER_H
#define ALCHEMY_NN_OPTIMIZER_SGD_MOMENTUM_OPTIMIZER_H

#include "nn/optimizer.h"

namespace alchemy {

template <typename T>
class MomentumOptimizer : public Optimizer<T> {
public:
    explicit MomentumOptimizer(const OptimizerParameter &param);
    virtual ~MomentumOptimizer() = default;

    virtual void optimize();
    virtual void update();

protected:
    vector<Tensor<T>> buf_;
    vector<Tensor<T>> buf2_;
};
}

#endif //!ALCHEMY_NN_OPTIMIZER_SGD_MOMENTUM_OPTIMIZER_H
