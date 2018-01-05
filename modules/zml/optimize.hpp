#ifndef _ZML_OPTIMIZE_HPP
#define _ZML_OPTIMIZE_HPP

//#include "network.hpp"
#include "layer.hpp"
#include "layer_factory.hpp"

namespace z {

class OptimizeParameter {
public:
    inline OptimizeParameter& epochs(int e) { epochs_ = e; return *this; }
    inline int epochs() const { return epochs_; }

    inline OptimizeParameter& train_net_param(const NetworkParameter& p) { train_net_param_ = p; train_net_param_.phase(TRAIN); return *this; }
    inline NetworkParameter train_net_param() const { return train_net_param_; }

    inline OptimizeParameter& test_net_param(const NetworkParameter& p) { test_net_param_ = p; test_net_param_.phase(TEST); return *this; }
    inline NetworkParameter test_net_param() const { return test_net_param_; }

private:
    int epochs_ = 0;
    NetworkParameter train_net_param_{};
    NetworkParameter test_net_param_{};
};

template <typename T>
class Optimize {
    using LayerType = Layer<T>;
public:
    explicit Optimize(const OptimizeParameter& param);
    void run();

private:
    OptimizeParameter param_;
    shared_ptr<Network<T>> train_net_;
    shared_ptr<Network<T>> test_net_;
};

template<typename T>
Optimize<T>::Optimize(const OptimizeParameter &param)
{
    param_ = param;

    LOG(INFO) << "====== TRAINING NETWORK ======";
    train_net_.reset(new Network<T>(param.train_net_param()));

    LOG(INFO) << "====== TEST NETWORK ======";
    test_net_.reset(new Network<T>(param.test_net_param()));
}

template<typename T>
void Optimize<T>::run()
{
    for(int i = 0; i < param_.epochs(); ++i) {
        train_net_->run();

        //
        test_net_->run();
        LOG(INFO) << "Epoch {" << i << "} :" <<  test_net_->accuracy();
    }
}

}

#endif //! _ZML_OPTIMIZE_HPP
