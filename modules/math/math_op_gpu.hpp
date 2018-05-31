namespace alchemy {

template <typename T>
__global__ void add_kernel(size_t size, const T *X1, const T *X2, T *Y)
{
    CUDA_FOREACH(size) {
        Y[idx] = X1[idx] + X2[idx];
    }
}

template <typename T>
void Add(const Tensor<GPU, T>& X1, const Tensor<GPU, T>& X2, Tensor<GPU, T>& Y)
{
    add_kernel<<<CUDA_BLOCK_NUM(X1.size()), CUDA_THREAD_NUM>>>(X1.size(), X1.ptr(), X2.ptr(), Y.mutable_ptr());
}
template <typename T>
__global__ void sub_kernel(size_t size, const T *X1, const T *X2, T *Y)
{
    CUDA_FOREACH(size) {
        Y[idx] = X1[idx] - X2[idx];
    }
}

template <typename T>
void Sub(const Tensor<GPU, T>& X1, const Tensor<GPU, T>& X2, Tensor<GPU, T>& Y)
{
    sub_kernel<<<CUDA_BLOCK_NUM(X1.size()), CUDA_THREAD_NUM>>>(X1.size(), X1.ptr(), X2.ptr(), Y.mutable_ptr());
}
template <typename T>
__global__ void div_kernel(size_t size, const T *X1, const T *X2, T *Y)
{
    CUDA_FOREACH(size) {
        Y[idx] = X1[idx] / X2[idx];
    }
}

template <typename T>
void Div(const Tensor<GPU, T>& X1, const Tensor<GPU, T>& X2, Tensor<GPU, T>& Y)
{
    div_kernel<<<CUDA_BLOCK_NUM(X1.size()), CUDA_THREAD_NUM>>>(X1.size(), X1.ptr(), X2.ptr(), Y.mutable_ptr());
}

template <typename T>
__global__ void mul_kernel(size_t size, const T *X1, const T *X2, T *Y)
{
    CUDA_FOREACH(size) {
        Y[idx] = X1[idx] * X2[idx];
    }
}

template <typename T>
void Mul(const Tensor<GPU, T>& X1, const Tensor<GPU, T>& X2, Tensor<GPU, T>& Y)
{ 
    mul_kernel<<<CUDA_BLOCK_NUM(X1.size()), CUDA_THREAD_NUM>>>(X1.size(), X1.gptr(), X2.gptr(), Y.mutable_gptr());
}

template <typename T>
__global__ void exp_kernel(size_t size, const T *X, T *Y)
{
    CUDA_FOREACH(size) {
        Y[idx] = std::exp(X[idx]);
    }
}

template <typename T>
void Exp(const Tensor<GPU, T>& X, Tensor<GPU, T>& Y)
{
    exp_kernel<<<CUDA_BLOCK_NUM(X.size()), CUDA_THREAD_NUM>>>(X.size(), X.gptr(), Y.mutable_gptr());
}

template <typename T>
__global__ void sign_kernel(size_t size, const T *X, T *Y)
{
    CUDA_FOREACH(size) {
        Y[idx] = (X[idx] > 0) - (0 > X[idx]);
    }
}

template <typename T>
void Sign(const Tensor<GPU, T>& X, Tensor<GPU, T>& Y)
{
    sign_kernel<<<CUDA_BLOCK_NUM(X.size()), CUDA_THREAD_NUM>>>(X.size(), X.gptr(), Y.mutable_gptr());
}
template <typename T>
// simoid
__global__ void sigmoid_kernel(size_t size, const T *X, T *Y)
{
    CUDA_FOREACH(size) {
        Y[idx] = 1.0 / (1.0 + std::exp(-X[idx]));
    }
}
template <typename T>
void Sigmoid(const Tensor<GPU, T>& X, Tensor<GPU, T>& Y)
{
    sigmoid_kernel<<<CUDA_BLOCK_NUM(X.size()), CUDA_THREAD_NUM>>>(X.size(), X.gptr(), Y.mutable_gptr());
}

template <typename T>
__global__ void sigmoid_grad_kernel(size_t size, const T* Y, const T *DY, T *DX)
{
    CUDA_FOREACH(size) {
        DX[idx] = DY[idx] * Y[idx] * (1.0 - Y[idx]);
    }
}

template <typename T>
void SigmoidGrad(const Tensor<GPU, T>& Y, const Tensor<GPU, T>& DY, Tensor<GPU, T>& DX)
{
    sigmoid_grad_kernel<<<CUDA_BLOCK_NUM(Y.size()), CUDA_THREAD_NUM>>>(Y.size(), Y.gptr(), DY.gptr(), DX.mutable_gptr());
}

// tanh
template <typename T>
__global__ void tanh_kernel(size_t size, const T *A, T *B)
{
    CUDA_FOREACH(size) {
        B[idx] = std::tanh(A[idx]);
    }
}
template <typename T>
void Tanh(const Tensor<GPU, T>& A, Tensor<GPU, T>& B)
{
    tanh_kernel<<<CUDA_BLOCK_NUM(A.size()), CUDA_THREAD_NUM>>>(A.size(), A.gptr(), B.mutable_gptr());
}

template <typename T>
__global__ void tanh_grad_kernel(size_t size, const T* A, const T *B, T *C)
{
    CUDA_FOREACH(size) {
        C[idx] = B[idx] * (1.0 - A[idx] * A[idx]);
    }
}

template <typename T>
void TanhGrad(const Tensor<GPU, T>& A, const Tensor<GPU, T>& B, Tensor<GPU, T>& C)
{
    tanh_grad_kernel<<<CUDA_BLOCK_NUM(A.size()), CUDA_THREAD_NUM>>>(A.size(), A.gptr(), B.gptr(), C.mutable_gptr());
}

// relu
template <typename T>
__global__ void relu_kernel(size_t size, const T *X, double alpha, T *Y)
{
    CUDA_FOREACH(size) {
        Y[idx] = X[idx] > 0 ? X[idx] : alpha * X[idx];
    }
}

template <typename T>
void ReLU(const Tensor<GPU, T>& X, double alpha, Tensor<GPU, T>& Y)
{
    relu_kernel<<<CUDA_BLOCK_NUM(X.size()), CUDA_THREAD_NUM>>>(X.size(), X.gptr(), alpha, Y.mutable_gptr());
}

template <typename T>
__global__ void relu_grad_kernel(size_t size, const T *X, const T *DY, double alpha, T *DX)
{
    CUDA_FOREACH(size) {
        DX[idx] = DY[idx] * (X[idx] > 0 ? 0 : alpha);
    }
}
template <typename T>
void ReLUGrad(const Tensor<GPU, T>& X, const Tensor<GPU, T>& DY, double alpha, Tensor<GPU, T>& DX)
{
    relu_grad_kernel<<<CUDA_BLOCK_NUM(X.size()), CUDA_THREAD_NUM>>>(X.size(), X.gptr(), DY.gptr(), alpha, DX.mutable_gptr());
}
}