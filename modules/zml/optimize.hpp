#ifndef _ZML_OPTIMIZE_HPP
#define _ZML_OPTIMIZE_HPP

#include "layer.hpp"
#include "layer_factory.hpp"

namespace z {

enum RegularizationType {
    L1, L2
};

class OptimizeParameter {
public:
    inline OptimizeParameter& epochs(int e) { epochs_ = e; return *this; }
    inline int epochs() const { return epochs_; }

    inline OptimizeParameter& train_net_param(const NetworkParameter& p) { train_net_param_ = p; train_net_param_.phase(TRAIN); return *this; }
    inline NetworkParameter train_net_param() const { return train_net_param_; }

    inline OptimizeParameter& test_net_param(const NetworkParameter& p) { test_net_param_ = p; test_net_param_.phase(TEST); return *this; }
    inline NetworkParameter test_net_param() const { return test_net_param_; }

    inline OptimizeParameter& weight_decay() { weight_decay_ = weight_decay_; return *this; }
    inline double weight_decay() const { return weight_decay_; }

    inline OptimizeParameter& test_iter(int ti) { test_iter_ = ti; return *this; }
    inline int test_iter() const { return test_iter_; }

    inline OptimizeParameter& max_iter(int mi) { max_iter_ = mi; return *this; }
    inline int max_iter() const { return max_iter_; }

    inline OptimizeParameter& test_interval(int ti) { test_interval_ = ti; return *this; }
    inline int test_interval() const { return test_interval_; }

    inline OptimizeParameter& regularization_type(RegularizationType type) { rt_ = type; return *this; }
    inline RegularizationType regularization_type() const { return rt_; }

private:
    int epochs_ = 0;
    double weight_decay_ = 0.0;

    int test_iter_ = 0;
    int max_iter_ = 0;
    int test_interval_ = 0;

    RegularizationType rt_ = L2;

    NetworkParameter train_net_param_{};
    NetworkParameter test_net_param_{};
};

template <typename T>
class Optimize {
    using LayerType = Layer<T>;
public:
    explicit Optimize(const OptimizeParameter& param);
    void run();

    void update();
    void regularize();
private:
    OptimizeParameter param_;

    vector<Tensor<T>> sign_;

    shared_ptr<Network<T>> net_;
    shared_ptr<Network<T>> test_net_;
};

template<typename T>
Optimize<T>::Optimize(const OptimizeParameter &param)
{
    param_ = param;

    LOG(INFO) << "====== TRAINING NETWORK ======";
    net_.reset(new Network<T>(param.train_net_param()));

    LOG(INFO) << "====== TEST NETWORK ======";
    test_net_.reset(new Network<T>(param.test_net_param()));

    /// setting
    const auto& lp = net_->learnable_params();
    for(const auto& param : lp) {
        sign_.push_back(Tensor<T>(std::get<0>(param)->shape()));
    }
}

template<typename T>
void Optimize<T>::run()
{
    for(auto iter = 0; iter < param_.max_iter(); ++iter) {
        net_->Forward();
        net_->Backward();

        regularize();
        update();

        if(iter && iter % param_.test_interval() == 0) {

            for(auto test_iter = 0; test_iter < param_.test_iter(); ++test_iter) {
                test_net_->Forward();
            }
            LOG(INFO) << "Iteration " << std::setw(6) << std::setfill(' ') << iter << " : " <<  test_net_->accuracy();
        }
    }
}

template<typename T>
void Optimize<T>::update()
{
    const auto& learnable_params = net_->learnable_params();
    for(auto& param : learnable_params) {
        vector_axpy(std::get<0>(param)->count(), (T)-std::get<1>(param), std::get<0>(param)->diff(), std::get<0>(param)->data());
    }
}

template<typename T>
void Optimize<T>::regularize()
{
    const auto& learnable_params = net_->learnable_params();

    switch(param_.regularization_type()) {
        case L1:
            for(size_t i = 0; i < learnable_params.size(); ++i) {
                const auto& item = learnable_params[i];
                vector_sign(std::get<0>(item)->count(),
                            std::get<0>(item)->data(),
                            sign_[i].data());

                vector_axpy(std::get<0>(item)->count(),
                            (T)(std::get<2>(item) * std::get<1>(item)),
                            sign_[i].data(),
                            std::get<0>(item)->diff());
            }
            break;

        case L2:
            for(auto& param : learnable_params) {
                vector_axpy(std::get<0>(param)->count(),
                            (T)(std::get<2>(param) * std::get<1>(param)),
                            std::get<0>(param)->data(),
                            std::get<0>(param)->diff());
            }
            break;
    }
}

}

#endif //! _ZML_OPTIMIZE_HPP
