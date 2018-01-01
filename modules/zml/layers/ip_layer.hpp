#ifndef _ZML_IP_LAYER_HPP
#define _ZML_IP_LAYER_HPP

extern "C" {
#include "cblas.h"
};
#include "zml/layer.hpp"
#include "zml/util/math_op.hpp"

namespace z {
template <typename T>
class InnerProductLayer: public Layer<T> {
    using container_type = Tensor<T>;
public:
    InnerProductLayer() = default;
    InnerProductLayer(int num, int chs, int rows, int cols);
    InnerProductLayer(const InnerProductLayer&)= delete;
    InnerProductLayer&operator=(const InnerProductLayer&)= delete;
    ~InnerProductLayer() = default;

    virtual void setup(const vector<container_type*>&input, const vector<container_type*>&output);

    virtual void ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardGPU(const vector<container_type*>& input, const vector<container_type*>& output);
#endif //! USE_CUDA

private:
    Tensor<T> weights_;
    Tensor<T> biases_;
    Tensor<T> biasmer_;

    int M_;
    int N_;
    int K_;
};

}


#endif //_ZML_IP_LAYER_HPP
