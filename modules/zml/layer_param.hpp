#ifndef _ZML_LAYER_PARAM_HPP
#define _ZML_LAYER_PARAM_HPP

#include <zml/util/filler.hpp>
#include "commen.hpp"
#include "zmatrix.h"

namespace z {

enum LayerType {
    ACCURACY_LAYER,
    EUCLIDEAN_LOSS_LAYER,
    INNER_PRODUCT_LAYER,
    INPUT_LAYER,
    RELU_LAYER,
    SIGMOID_CROSS_ENTORPY_LOSS_LAYER,
    SIGMOID_LAYER,
    SOFTMAX_LAYER,
    TANH_LAYER
};

class AccuracyParameter {};

class EuclideanLossParameter {};

class InnerProductParameter {
public:
    /// Neuron size
    inline InnerProductParameter& neuron_size(size_t size) { neuron_size_ = size; return *this; }
    inline size_t neuron_size() const { return neuron_size_; }

    /// init
    inline InnerProductParameter& weight_filler(FillerType filler) { weight_filler_ = filler; return *this; }
    inline FillerType weight_filler() const { return weight_filler_; }
    inline InnerProductParameter& bias_filler(FillerType filler) { bias_filler_ = filler; return *this; }
    inline FillerType bias_filler() const { return bias_filler_; }

    /// learning rate
    inline InnerProductParameter& wlr(double lr) { wlr_ = lr; return *this; }
    inline double wlr() const { return wlr_; }

    inline InnerProductParameter& blr(double lr) { blr_ = lr; return *this; }
    inline double blr() const { return blr_; }

private:
    size_t neuron_size_ = 1;
    FillerType weight_filler_ = NORMAL;
    FillerType bias_filler_ = NORMAL;
    double wlr_ = 1.0;
    double blr_ = 1.0;
};

class InputParameter {
public:
    /// batch size
    InputParameter& batch_size(size_t size) { batch_size_ = size; return *this; }
    size_t batch_size() const { return batch_size_; }

    /// data
    InputParameter& source(MnistLoader& loader) { data_ = loader.data(); return *this;}
    vector<pair<Matrix, uint8_t>> data() const { return data_; };

    /// scale
    InputParameter& scale(double scale) { scale_ = scale; return *this; }
    double scale() const { return scale_; }

private:
    vector<pair<Matrix, uint8_t>> data_{};

    size_t batch_size_ = 1;
    double scale_ = 1.0;
};

class ReLuParameter{
public:
    inline ReLuParameter& alpha(double a) { alpha_ = a; return *this; }
    inline double alpha() const { return alpha_; }
private:
    double alpha_ = 0.;
};

class SigmoidCrossEntropyLossParameter{};

class SigmoidParameter {};

class SoftmaxParameter{};

class TanhParameter {};

class LayerParameter {
public:
    // phase
    inline LayerParameter& phase(Phase p) { phase_ = p; return *this; }
    inline Phase phase() const { return phase_; }

    // name
    inline LayerParameter& name(const string& name) { name_ = name; return *this; }
    inline string name() const { return name_; }

    // type
    inline LayerParameter& type(LayerType type) { type_ = type; return *this; }
    inline LayerType type() const { return type_; }

    // input
    inline LayerParameter& input(const string& in) { inputs_.push_back(in); return *this; }
    inline vector<string> inputs() const { return inputs_; }

    // output
    inline LayerParameter& output(const string& out) { outputs_.push_back(out); return *this; }
    inline vector<string> outputs() const { return outputs_; }

    //
    inline LayerParameter& input_param(const InputParameter& in_param) { input_param_ = in_param; return *this; }
    inline InputParameter input_param() const { return input_param_; }

    //
    inline LayerParameter& ip_param(const InnerProductParameter& ip_param) { ip_param_ = ip_param; return *this; }
    inline InnerProductParameter ip_param() const { return ip_param_; }

    inline LayerParameter& sigmoid_param(const SigmoidParameter& sp) { sigmoid_param_ = sp; return *this; }
    inline SigmoidParameter sigmoid_param() const { return sigmoid_param_; }

    inline LayerParameter& euclidean_param(const EuclideanLossParameter& ep) { euclidean_param_ = ep; return *this; }
    inline EuclideanLossParameter euclidean_param() const { return euclidean_param_; }

    inline LayerParameter& accuracy_param(const AccuracyParameter& ap) { accuracy_param_ = ap; return *this; }
    inline AccuracyParameter accuracy_param() const { return accuracy_param_; }

    inline LayerParameter& sigmoid_cross_entropy_loss_param(const SigmoidCrossEntropyLossParameter &cp) { cross_entropy_loss_param_ = cp; return *this; }
    inline SigmoidCrossEntropyLossParameter sigmoid_cross_entropy_loss_param() const { return cross_entropy_loss_param_; }

    inline LayerParameter& tanh_param(const TanhParameter& tp) { tanh_param_ = tp; return *this; }
    inline TanhParameter tanh_param() const { return tanh_param_; }

    inline LayerParameter& softmax_param(const SoftmaxParameter& sp) { softmax_param_ = sp; return *this; }
    inline SoftmaxParameter softmax_param() const { return softmax_param_; }

    inline LayerParameter& relu_param(const ReLuParameter& rp) { relu_param_ = rp; return *this; }
    inline ReLuParameter relu_param() const { return relu_param_; }

protected:
    Phase phase_ = DEFAULT;
    string name_{};
    LayerType type_ = INNER_PRODUCT_LAYER;

    vector<string> inputs_{};
    vector<string> outputs_{};

    /// layers
    AccuracyParameter accuracy_param_;
    SigmoidCrossEntropyLossParameter cross_entropy_loss_param_;
    EuclideanLossParameter euclidean_param_;
    InnerProductParameter ip_param_;
    InputParameter input_param_;
    ReLuParameter relu_param_;
    SigmoidParameter sigmoid_param_;
    SoftmaxParameter softmax_param_;
    TanhParameter tanh_param_;
};


}

#endif //! _ZML_LAYER_PARAM_HPP
