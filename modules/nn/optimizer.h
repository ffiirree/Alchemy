#ifndef ALCHEMY_NN_OPTIMIZE_H
#define ALCHEMY_NN_OPTIMIZE_H

#include "layer.h"
#include "network.h"
#include "layer_factory.h"
#include "math/math_op.h"

namespace alchemy {

enum RegularizationType {
    L1, L2
};

class OptimizerParameter {
public:
    inline OptimizerParameter& epochs(int e) { epochs_ = e; return *this; }
    inline int epochs() const { return epochs_; }

    inline OptimizerParameter& train_net_param(const NetworkParameter& p) { train_net_param_ = p; train_net_param_.phase(TRAIN); return *this; }
    inline NetworkParameter train_net_param() const { return train_net_param_; }

    inline OptimizerParameter& test_net_param(const NetworkParameter& p) { test_net_param_ = p; test_net_param_.phase(TEST); return *this; }
    inline NetworkParameter test_net_param() const { return test_net_param_; }

    inline OptimizerParameter& weight_decay() { weight_decay_ = weight_decay_; return *this; }
    inline double weight_decay() const { return weight_decay_; }

    inline OptimizerParameter& test_iter(int ti) { test_iter_ = ti; return *this; }
    inline int test_iter() const { return test_iter_; }

    inline OptimizerParameter& max_iter(int mi) { max_iter_ = mi; return *this; }
    inline int max_iter() const { return max_iter_; }

    inline OptimizerParameter& test_interval(int ti) { test_interval_ = ti; return *this; }
    inline int test_interval() const { return test_interval_; }

    inline OptimizerParameter& regularization_type(RegularizationType type) { rt_ = type; return *this; }
    inline RegularizationType regularization_type() const { return rt_; }

    inline OptimizerParameter& mode(Global::Mode md) { Global::mode(md); return *this; }
    inline Global::Mode mode() const { return Global::mode(); }

    inline OptimizerParameter& momentum(double m) { momentum_ = m; return *this; }
    inline double momentum() const { return momentum_; }

private:

    int epochs_ = 0;
    double weight_decay_ = 0.0;

    int test_iter_ = 0;
    int max_iter_ = 0;
    int test_interval_ = 0;

    double momentum_ = 0.0;

    RegularizationType rt_ = L2;

    NetworkParameter train_net_param_{};
    NetworkParameter test_net_param_{};
};

template <typename Device, typename T>
class Optimizer {
    using LayerType = Layer<Device, T>;
public:
    explicit Optimizer(const OptimizerParameter& param);
    virtual ~Optimizer() = default;

    void regularize();

    virtual void optimize() = 0;
    virtual void update() = 0;

    void save(string path) { net_->save(path); }
    void load(string path) { net_->load(path); }

protected:
    OptimizerParameter param_;

    vector<Blob<Device, T>> sign_;

    shared_ptr<Network<Device, T>> net_;
    shared_ptr<Network<Device, T>> test_net_;
};

template <typename Device, typename T>
Optimizer<Device, T>::Optimizer(const OptimizerParameter &param)
{
    param_ = param;

    LOG(INFO) << "Mode: " << ((Global::mode() == Global::Mode::CPU) ? "CPU": "GPU");

    LOG(INFO) << "========= TRAINING NETWORK =========";
    net_.reset(new Network<Device, T>(param.train_net_param()));

    LOG(INFO) << "========= TEST NETWORK =========";
    test_net_.reset(new Network<Device, T>(param.test_net_param()));

    /// setting
    const auto& lp = net_->learnable_params();
    for(const auto& _param : lp) {
        sign_.push_back(Blob<Device, T>(std::get<0>(_param)->shape()));
    }
}

template <typename Device, typename T>
void Optimizer<Device, T>::regularize()
{
    const auto& learnable_params = net_->learnable_params();

    switch(param_.regularization_type()) {
        case L1:
            for(size_t i = 0; i < learnable_params.size(); ++i) {
                const auto& item = learnable_params[i];

                if(Global::mode() == Global::CPU) {
                    vector_sign(std::get<0>(item)->size(),
                                std::get<0>(item)->data_cptr(),
                                sign_[i].mutable_data_cptr());

                    vector_axpy(std::get<0>(item)->size(),
                                (T)(std::get<2>(item) * std::get<1>(item)),
                                sign_[i].data_cptr(),
                                std::get<0>(item)->mutable_diff_cptr());
                }
                else {
                    vector_sign_gpu(std::get<0>(item)->size(),
                                    std::get<0>(item)->data_gptr(),
                                    sign_[i].mutable_data_gptr());

                    vector_axpy_gpu(std::get<0>(item)->size(),
                                    (T)(std::get<2>(item) * std::get<1>(item)),
                                    sign_[i].data_gptr(),
                                    std::get<0>(item)->mutable_diff_gptr());
                }
            }
            break;

        case L2:
            for(auto& param : learnable_params) {
                if(Global::mode() == Global::CPU) {
                    vector_axpy(std::get<0>(param)->size(),
                                (T)(std::get<2>(param) * std::get<1>(param)),
                                std::get<0>(param)->data_cptr(),
                                std::get<0>(param)->mutable_diff_cptr());
                }
                else {
                    vector_axpy_gpu(std::get<0>(param)->size(),
                                    (T)(std::get<2>(param) * std::get<1>(param)),
                                    std::get<0>(param)->data_gptr(),
                                    std::get<0>(param)->mutable_diff_gptr());
                }
            }
            break;
    }
}
}

#endif //! ALCHEMY_NN_OPTIMIZE_H
