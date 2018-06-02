#ifndef ALCHEMY_NN_NETWORK_H
#define ALCHEMY_NN_NETWORK_H

#include <fstream>
#include <glog/logging.h>
#include "nn/blob.h"
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

    inline vector<LayerParameter> filter(Phase phase) const;
private:
    string name_{};
    Phase phase_ = TRAIN;
    vector<LayerParameter> layer_params_{};
};

vector<LayerParameter> NetworkParameter::filter(Phase phase) const
{
    vector<LayerParameter> params;
    for(const auto& param : layer_params_) {
        if(param.phase() == phase || param.phase() == SHARED) {
            params.push_back(param);
        }
    }
    return params;
}

template <typename Device, typename T>
class Network {
public:
    Network()= default;
    explicit Network(const NetworkParameter& net_params);
    ~Network() = default;

    double accuracy() { return outputs_.back()[0]->data_cptr()[0]; }
    double loss()
    {
//        LOG(INFO) << input_.back()[0]->data_cptr()[0] << " " << input_.back()[1]->data_cptr()[0] << " [" << std::fabs(input_.back()[0]->data_cptr()[0] - input_.back()[1]->data_cptr()[0]) << "]";
        return outputs_.back()[0]->data_cptr()[0];
    }

    inline vector<tuple<shared_ptr<Blob<Device, T>>, double, double>> learnable_params() const { return learnable_params_; };

    inline vector<shared_ptr<Layer<Device, T>>> layers() const { return layers_; }
    inline vector<vector<Blob<Device, T>*>> inputs() const { return inputs_; }
    inline vector<vector<Blob<Device, T>*>> outputs() const { return outputs_; }

    inline Phase phase() const { return phase_; }

    void Forward();
    void Backward();

    void save(string path);
    void load(string path);

private:
    Phase phase_ = SHARED;

    // 层
    vector<shared_ptr<Layer<Device, T>>> layers_{};
    // 从输入到输出的数据
    map<string, shared_ptr<Blob<Device, T>>> data_flow_{};
    // 参数
    vector<tuple<shared_ptr<Blob<Device, T>>, double, double>> learnable_params_{};

    // 和每一层一一对应
    vector<vector<Blob<Device, T>*>> inputs_{};
    vector<vector<Blob<Device, T>*>> outputs_{};
};


template <typename Device, typename T>
void Network<Device, T>::Forward()
{
    for(size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
        layers_[layer_index]->Forward(inputs_[layer_index], outputs_[layer_index]);
    }
}

template <typename Device, typename T>
void Network<Device, T>::Backward()
{
    for(size_t layer_index = layers_.size(); layer_index > 0; --layer_index) {
        layers_[layer_index - 1]->Backward(inputs_[layer_index - 1], outputs_[layer_index - 1]);
    }
}

template <typename Device, typename T>
Network<Device, T>::Network(const NetworkParameter &net_params)
{
    phase_ = net_params.phase();
    auto layer_params = net_params.filter(phase_);

    inputs_.resize(layer_params.size());
    outputs_.resize(layer_params.size());

    for(size_t layer_idx = 0; layer_idx < layer_params.size(); ++layer_idx) {
        const auto param = layer_params[layer_idx];

        // Create layers
        LOG(INFO) << "Creating Layer: " << param.name();
        layers_.push_back(LayerFactory<Device, T>::GetLayer(param));

        // inputs
        for(const auto& input: param.inputs()) {
            auto search = data_flow_.find(input);
            if(search == data_flow_.end()) {
                data_flow_[input] = shared_ptr<Blob<Device, T>>(new Blob<Device, T>());
            }
            LOG(INFO) << param.name() << " <-- " << input;
            inputs_[layer_idx].push_back(data_flow_[input].get());
        }

        // outputs
        for(const auto& output: param.outputs()) {
            auto search = data_flow_.find(output);
            if(search == data_flow_.end()) {
                data_flow_[output] = shared_ptr<Blob<Device, T>>(new Blob<Device, T>());
            }
            LOG(INFO) << param.name() << " --> " << output;
            outputs_[layer_idx].push_back(data_flow_[output].get());
        }

        // Set up
        LOG(INFO) << "Setting up: " << param.name();
        for(size_t i = 0; i < inputs_[layer_idx].size(); ++i) {
            LOG(INFO) << "input  #" << std::to_string(i) << ": " << inputs_[layer_idx][i]->shape();
        }
        layers_[layer_idx]->setup(inputs_[layer_idx], outputs_[layer_idx]);
        for(size_t i = 0; i < outputs_[layer_idx].size(); ++i) {
            LOG(INFO) << "output #" << std::to_string(i) << ": " << outputs_[layer_idx][i]->shape();
        }

        // Learnable parameters
        const auto& lp = layers_[layer_idx]->learnable_params();
        learnable_params_.insert(learnable_params_.end(), lp.begin(), lp.end());
    }
}

template <typename Device, typename T>
void Network<Device, T>::save(string path)
{
    std::fstream file(path, std::ios::out | std::ios::binary);

    if(!file.is_open()) LOG(FATAL) << "Open file failure.";

    for(const auto& param : learnable_params_) {
        auto blob = std::get<0>(param);
        auto data = blob->data_cptr();
        auto total = blob->size();
        file.write(reinterpret_cast<const char*>(data), total * sizeof(T));
    }
    file.close();
}

template <typename Device, typename T>
void Network<Device, T>::load(string path)
{
    std::fstream file(path, std::ios::in | std::ios::binary);

    if(!file.is_open()) LOG(FATAL) << "Open file failure.";

    for(auto& param : learnable_params_) {
        auto blob = std::get<0>(param);
        auto data = blob->mutable_data_cptr();
        auto total = blob->size();
        file.read(reinterpret_cast<char*>(data), total * sizeof(T));
    }
    file.close();
}
}

#endif //! ALCHEMY_NN_NETWORK_H
