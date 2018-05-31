namespace alchemy {

template <typename T>
void Add(const Tensor<CPU, T>& X1, const Tensor<CPU, T>& X2, Tensor<CPU, T>& Y)
{
    for(size_t idx = 0; idx < X1.size(); ++idx) {
        Y.at(idx) = X1.at(idx) + X2.at(idx);
    }
}
template <typename T>
void Sub(const Tensor<CPU, T>& X1, const Tensor<CPU, T>& X2, Tensor<CPU, T>& Y)
{
    for(size_t idx = 0; idx < X1.size(); ++idx) {
        Y.at(idx) = X1.at(idx) - X2.at(idx);
    }
}

template <typename T>
void Mul(const Tensor<CPU, T>& X1, const Tensor<CPU, T>& X2, Tensor<CPU, T>& Y)
{
    for(size_t idx = 0; idx < X1.size(); ++idx) {
        Y.at(idx) = X1.at(idx) * X2.at(idx);
    }
}

template <typename T>
void Div(const Tensor<CPU, T>& X1, const Tensor<CPU, T>& X2, Tensor<CPU, T>& Y)
{
    for(size_t idx = 0; idx < X1.size(); ++idx) {
        Y.at(idx) = X1.at(idx) / X2.at(idx);
    }
}

template <typename T>
void Exp(const Tensor<CPU, T>& X, Tensor<CPU, T>& Y)
{
    for(size_t idx = 0; idx < X.size(); ++idx) {
        Y.at(idx) = std::exp(X.at(idx));
    }
}
template <typename T>
void Sign(const Tensor<CPU, T>& X, Tensor<CPU, T>& Y)
{
    for(size_t idx = 0; idx < X.size(); ++idx) {
        Y.at(idx) = (X.at(idx) > 0) - (0 > X.at(idx));
    }
}
template <typename T>
void Sigmoid(const Tensor<CPU, T>& X, Tensor<CPU, T>& Y)
{
    for(size_t idx = 0; idx < X.size(); ++idx) {
        Y.at(idx) = 1.0 / (1.0 + std::exp(-X.at(idx)));
    }
}

template <typename T>
void SigmoidGrad(const Tensor<CPU, T>& Y, const Tensor<CPU, T>& DY, Tensor<CPU, T>& DX)
{
    for(size_t idx = 0; idx < Y.size(); ++idx) {
        DX.at(idx) = DY.at(idx) * Y.at(idx) * (1.0 - Y.at(idx));
    }
}

template <typename T>
void Tanh(const Tensor<CPU, T>& X, Tensor<CPU, T>& Y)
{
    for(size_t idx = 0; idx < X.size(); ++idx) {
        Y.at(idx) = std::tanh(X.at(idx));
    }
}

template <typename T>
void TanhGrad(const Tensor<CPU, T>& Y, const Tensor<CPU, T>& DY, Tensor<CPU, T>& DX)
{
    for(size_t idx = 0; idx < Y.size(); ++idx) {
        DX.at(idx) = DY.at(idx) * (1.0 - Y.at(idx) * Y.at(idx));
    }
}

template <typename T>
void ReLU(const Tensor<CPU, T>& X, double alpha, Tensor<CPU, T>& Y)
{
    for(size_t idx = 0; idx < X.size(); ++idx) {
        Y.at(idx) = X.at(0) > 0 ? X.at(idx) : alpha * X.at(idx);
    }
}
template <typename T>
void ReLUGrad(const Tensor<CPU, T>& X, const Tensor<CPU, T>& DY, double alpha, Tensor<CPU, T>& DX)
{
    for(size_t idx = 0; idx < X.size(); ++idx) {
        DX.at(idx) = DY.at(idx) * (X.at(idx) > 0 ? 1 : alpha);
    }
}
}