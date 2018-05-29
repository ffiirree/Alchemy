#ifndef ALCHEMY_MATH_MATH_OP_H
#define ALCHEMY_MATH_MATH_OP_H

#include <util/util.h>
#include <cmath>
#include "core/tensor.h"

extern "C"{
#include "cblas.h"
};


namespace alchemy {

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

// Y = alpha * X + Y
template <typename T> void vector_axpy(const int count, const T alpha, const T* X, T *Y);
template <typename T> void vector_axpy_gpu(const int count, const T alpha, const T* X, T *Y);
// X = alpha * X
template <typename T> void vector_scal(const int count, const T alpha, T* X);
template <typename T> void vector_scal_gpu(const int count, const T alpha, T* X);
// Y = X
template <typename T> void vector_copy(const int count, const T* X, T* Y);
template <typename T> void vector_copy_gpu(const int count, const T* X, T* Y);
// foreach Xi = value
template <typename T> void vector_set(const int count, const T value, T* X);
template <typename T> void vector_set_gpu(const int count, const T value, T* X);
// C = A - B
template <typename T> void vector_sub(const int count, const T* A, const T* B, T* C);
template <typename T> void vector_sub_gpu(const int count, const T* A, const T* B, T* C);
// C = A + B
template <typename T> void vector_add(const int count, const T* A, const T* B, T* C);
template <typename T> void vector_add_gpu(const int count, const T* A, const T* B, T* C);
// return t(A)B
template <typename T> T    vector_dot(const int count, const T* A, const T* B);
template <typename T> T    vector_dot_gpu(const int count, const T* A, const T* B);
// B = exp(A)
template <typename T> void vector_exp(const int count, const T* A, T* B);
template <typename T> void vector_exp_gpu(const int count, const T* A, T* B);
//
template <typename T> void vector_div(const int count, const T* A, const T* B, T* C);
template <typename T> void vector_div_gpu(const int count, const T* A, const T* B, T* C);
//
template <typename T> void vector_sign(const int count, const T* A, T* B);
template <typename T> void vector_sign_gpu(const int count, const T* A, T* B);

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

template <typename T>
void matvec_mul(const enum CBLAS_TRANSPOSE TransA,
                const int M, const int N,
                const T alpha, const T *A, const T *X,
                const T beta, T *Y);
template <typename T>
void matvec_mul_gpu(const enum CBLAS_TRANSPOSE TransA,
                    const int M, const int N,
                    const T alpha, const T *A, const T *X,
                    const T beta, T *Y);


/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */
// C = alpha * op(A)op(B) + beta * C
template <typename T>
void matrix_mul(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                const int M, const int N, const int K,
                const T alpha, const T *A, const T *B,
                const T beta, T *C);

template <typename T>
void matrix_mul_gpu(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                    const int M, const int N, const int K,
                    const T alpha, const T *A, const T *B,
                    const T beta, T *C);


/// 
template <typename T> void print_cpu(const int count, const T* A);
template <typename T> void print_gpu(const int count, const T* A);

template <typename T>
void Sigmoid(const Tensor<CPU, T>& A, Tensor<CPU, T>& B)
{
    for(size_t idx = 0; idx < A.size(); ++idx) {
        B.cat(idx) = 1.0 / (1.0 + std::exp(-A.cat(idx)));
    }
}
#ifdef __CUDACC__
template <typename T>
void Sigmoid(const Tensor<GPU, T>& A, Tensor<GPU, T>& B)
{

}
#endif
template <typename T>
void SigmoidGrad(const Tensor<CPU, T>& A, const Tensor<CPU, T>& B, Tensor<CPU, T>& C)
{
    for(size_t idx = 0; idx < A.size(); ++idx) {
        C.cat(idx) = B.cat(idx) * A.cat(idx) * (1.0 - A.cat(idx));
    }
}
#ifdef __CUDACC__
template <typename T>
void SigmoidGrad(const Tensor<GPU, T>& A, const Tensor<GPU, T>& B, Tensor<GPU, T>& C)
{

}
#endif
template <typename T> void Tanh(Tensor<CPU, T>& A, Tensor<CPU, T>& B);
template <typename T> void Tanh(Tensor<GPU, T>& A, Tensor<GPU, T>& B);

template <typename T> void TanhGrad(Tensor<CPU, T>& A, Tensor<CPU, T>& B);
template <typename T> void TanhGrad(Tensor<GPU, T>& A, Tensor<GPU, T>& B);
}
#endif //! ALCHEMY_MATH_MATH_OP_H
