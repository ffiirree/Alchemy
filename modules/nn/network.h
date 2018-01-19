#ifndef ALCHEMY_NN_NETWORK_H
#define ALCHEMY_NN_NETWORK_H

#include <glog/logging.h>
#include "core/tensor.h"
#include "layer_factory.h"

namespace alchemy {

class NetworkParameter{
public:
    NetworkParameter() = default;

    explicit NetworkParameter(const vector<LayerParameter>& params) : layer_params_(params) { }

    inline NetworkParameter& name(const string& name) { name_ = name; return *this; }
    inline string name() const { return name_; }

    inline NetworkParameter& phase(Phase p) { phase_ = p; return *this; }
    inline Phase phase() const { return phase_; }

    inline NetworkParameter& layer_params(const vector<LayerParameter>& params) { layer_params_ = params; return *this; }
    inline vector<LayerParameter> layer_params() const { return layer_params_; }

private:
    string name_{};
    Phase phase_ = TRAIN;
    vector<LayerParameter> layer_params_{};
};

template <typename T>
class Network {
public:
    Network()= default;
    explicit Network(const NetworkParameter& param);
    ~Network() = default;

    double accuracy() { return output_.back()[0]->cpu_data()[0]; }

    inline vector<tuple<shared_ptr<Tensor<T>>, double, double>> learnable_params() const { return learnable_params_; };

    inline vector<shared_ptr<Layer<T>>> layers() const { return layers_; }
    inline vector<vector<Tensor<T>*>> inputs() const { return input_; }
    inline vector<vector<Tensor<T>*>> outputs() const { return output_; }

    inline Phase phase() const { return phase_; }

    void Forward();
    void Backward();

private:
    Phase phase_ = DEFAULT;

    // 层
    vector<shared_ptr<Layer<T>>> layers_{};
    // 从输入到输出的数据
    map<string, shared_ptr<Tensor<T>>> data_flow_{};
    // 参数
    vector<tuple<shared_ptr<Tensor<T>>, double, double>> learnable_params_{};

    // 和每一层一一对应
    vector<vector<Tensor<T>*>> input_{};
    vector<vector<Tensor<T>*>> output_{};
};


template<typename T>
void Network<T>::Forward()
{
    for(size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
        layers_[layer_index]->Forward(input_[layer_index], output_[layer_index]);
    }
}

template<typename T>
void Network<T>::Backward()
{
    for(size_t layer_index = layers_.size(); layer_index > 0; --layer_index) {
        layers_[layer_index - 1]->Backward(input_[layer_index - 1], output_[layer_index - 1]);
    }
}

template<typename T>
Network<T>::Network(const NetworkParameter &param)
{
    phase_ = param.phase();
    for(auto& layer_param: param.layer_params()) {
        if(phase_ == layer_param.phase() || layer_param.phase() == DEFAULT) {
            layers_.push_back(LayerFactory<T>::GetLayer(layer_param.phase(phase_)));
        }
    }

    input_.resize(layers_.size());
    output_.resize(layers_.size());

    for(size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {

        /// inputs
        for(auto& input_name : layers_[layer_index]->parameter().inputs()) {
            auto search = data_flow_.find(input_name);
            if(search == data_flow_.end()) {
                data_flow_[input_name] = shared_ptr<Tensor<T>>(new Tensor<T>());
            }
            input_[layer_index].push_back(data_flow_[input_name].get());
        }

        /// outputs
        for(auto& output_name : layers_[layer_index]->parameter().outputs()) {
            auto search = data_flow_.find(output_name);
            if(search == data_flow_.end()) {
                data_flow_[output_name] = shared_ptr<Tensor<T>>(new Tensor<T>());
            }
            output_[layer_index].push_back(data_flow_[output_name].get());
        }

        layers_[layer_index]->setup(input_[layer_index], output_[layer_index]);

        const auto& lp = layers_[layer_index]->learnable_params();
        learnable_params_.insert(learnable_params_.end(), lp.begin(), lp.end());
    }
}
}

#endif //! ALCHEMY_NN_NETWORK_H
