#ifndef _ZML_SIGMOID_LAYER_HPP
#define _ZML_SIGMOID_LAYER_HPP

#include "zml/layer.hpp"

namespace z {
template <typename T>
class SigmoidLayer: public Layer<T> {
    using container_type = Tensor<T>;
public:
    SigmoidLayer() = default;
    SigmoidLayer(int num, int chs, int rows, int cols);
    SigmoidLayer(const SigmoidLayer&)= delete;
    SigmoidLayer&operator=(const SigmoidLayer&)= delete;
    ~SigmoidLayer() = default;

    virtual void setup(const vector<container_type*>&input, const vector<container_type*>&output);

    virtual void ForwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardCPU(const vector<container_type*>& input, const vector<container_type*>& output);

#ifdef USE_CUDA
    virtual void ForwardGPU(const vector<container_type*>& input, const vector<container_type*>& output);
    virtual void BackwardGPU(const vector<container_type*>& input, const vector<container_type*>& output);
#endif //! USE_CUDA
};


}

#endif //_ZML_SIGMOID_LAYER_HPP
