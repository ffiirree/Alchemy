#ifndef _ZML_SGD_OPTIMIZER_HPP
#define _ZML_SGD_OPTIMIZER_HPP

#include <zml/optimizer.hpp>

namespace z {

template <typename T>
class SgdOptimizer : public Optimizer<T> {
public:
    explicit SgdOptimizer(const OptimizerParameter &param) : Optimizer<T>(param) {}
    virtual ~SgdOptimizer() = default;

    virtual void optimize();
    virtual void update();
};

}

#endif //! _ZML_SGD_OPTIMIZER_HPP
