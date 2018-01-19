#ifndef ALCHEMY_NN_OPTIMIZER_SGD_OPTIMIZER_H
#define ALCHEMY_NN_OPTIMIZER_SGD_OPTIMIZER_H

#include "nn/optimizer.h"

namespace alchemy {

template <typename T>
class SgdOptimizer : public Optimizer<T> {
public:
    explicit SgdOptimizer(const OptimizerParameter &param) : Optimizer<T>(param) {}
    virtual ~SgdOptimizer() = default;

    virtual void optimize();
    virtual void update();
};
}

#endif //! ALCHEMY_NN_OPTIMIZER_SGD_OPTIMIZER_H
