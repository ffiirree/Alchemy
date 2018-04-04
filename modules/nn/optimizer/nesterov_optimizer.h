#ifndef ALCHEMY_NN_OPTIMIZER_NESTEROV_OPTIMIZER_H
#define ALCHEMY_NN_OPTIMIZER_NESTEROV_OPTIMIZER_H

#include "nn/optimizer.h"

namespace alchemy {
template <typename T>
class NesterovOptimizer : public Optimizer<T> {
public:
    explicit NesterovOptimizer(const OptimizerParameter &param);
    virtual ~NesterovOptimizer() = default;

    virtual void optimize();
    virtual void update();

protected:
    vector<Tensor<T>> buf_;
    vector<Tensor<T>> buf2_;
};
}

#endif //!ALCHEMY_NN_OPTIMIZER_NESTEROV_OPTIMIZER_H
