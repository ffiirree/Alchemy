namespace alchemy {

template <typename T>
void Sigmoid(const Tensor<CPU, T>& X, Tensor<CPU, T>& Y)
{
    for(size_t idx = 0; idx < X.size(); ++idx) {
        Y.cat(idx) = 1.0 / (1.0 + std::exp(-X.cat(idx)));
    }
}

template <typename T>
void SigmoidGrad(const Tensor<CPU, T>& Y, const Tensor<CPU, T>& DY, Tensor<CPU, T>& DX)
{
    for(size_t idx = 0; idx < Y.size(); ++idx) {
        DX.cat(idx) = DY.cat(idx) * Y.cat(idx) * (1.0 - Y.cat(idx));
    }
}

template <typename T>
void Tanh(const Tensor<CPU, T>& X, Tensor<CPU, T>& Y)
{
    for(size_t idx = 0; idx < X.size(); ++idx) {
        Y.cat(idx) = std::tanh(X.cat(idx));
    }
}

template <typename T>
void TanhGrad(const Tensor<CPU, T>& Y, const Tensor<CPU, T>& DY, Tensor<CPU, T>& DX)
{
    for(size_t idx = 0; idx < Y.size(); ++idx) {
        DX.cat(idx) = DY.cat(idx) * (1.0 - Y.cat(idx) * Y.cat(idx));
    }
}

template <typename T>
void ReLU(const Tensor<CPU, T>& X, double alpha, Tensor<CPU, T>& Y)
{
    for(size_t idx = 0; idx < X.size(); ++idx) {
        Y.cat(idx) = X.cat(0) > 0 ? X.cat(idx) : alpha * X.cat(idx);
    }
}
template <typename T>
void ReLUGrad(const Tensor<CPU, T>& X, const Tensor<CPU, T>& DY, double alpha, Tensor<CPU, T>& DX)
{
    for(size_t idx = 0; idx < X.size(); ++idx) {
        DX.cat(idx) = DY.cat(idx) * (X.cat(idx) > 0 ? 1 : alpha);
    }
}
}