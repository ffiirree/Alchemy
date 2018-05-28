#ifndef ALCHEMY_NN_LAYERS_IP_LAYER_H
#define ALCHEMY_NN_LAYERS_IP_LAYER_H

#include "nn/layer.h"

namespace alchemy {

template <typename T>
class InnerProductLayer: public Layer<T> {
public:
    using container = Blob<T>;

    InnerProductLayer() = default;
    explicit InnerProductLayer(const LayerParameter& parameter)
            : Layer<T>(parameter), ip_param_(parameter.ip_param()),
              weights_(new Blob<T>()), biases_(new Blob<T>()) { }
    virtual ~InnerProductLayer() = default;

    void setup(const vector<container *>&input, const vector<container *>&output) override;

    void ForwardCPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardCPU(const vector<container *>& input, const vector<container *>& output) override;

#ifdef __CUDACC__
    void ForwardGPU(const vector<container *>& input, const vector<container *>& output) override;
    void BackwardGPU(const vector<container *>& input, const vector<container *>& output) override;
#endif //! __CUDACC__

private:
    InnerProductParameter ip_param_{};

    shared_ptr<Blob<T>> weights_;
    shared_ptr<Blob<T>> biases_;
    Tensor<T> biasmer_;

    int M_ = 0;
    int N_ = 0;
    int K_ = 0;
};
} // namespace

#include "inner_product_layer.hpp"
#ifdef __CUDACC__
#include "inner_product_layer.cuh"
#endif //! __CUDACC__
#endif //! ALCHEMY_NN_LAYERS_IP_LAYER_H
