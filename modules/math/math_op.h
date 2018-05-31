#ifndef ALCHEMY_MATH_MATH_OP_H
#define ALCHEMY_MATH_MATH_OP_H

#include <cmath>
#include "util/util.h"
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
//template <typename T> void vector_axpy(const int count, const T alpha, const T* X, T *Y);
//template <typename T> void vector_axpy_gpu(const int count, const T alpha, const T* X, T *Y);

template <typename T> void axpy(T alpha, const Tensor<CPU, T> &X, Tensor<CPU, T> &Y);
template <typename T> void axpy(T alpha, const Tensor<GPU, T> &X, Tensor<GPU, T> &Y);

//// X = alpha * X
template <typename T> void Scale(const int count, const T alpha, T* X);


// return t(A)B
//template <typename T> T    vector_dot(const int count, const T* A, const T* B);
//template <typename T> T    vector_dot_gpu(const int count, const T* A, const T* B);


/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */
//
template <typename Device, typename T>
void matvec_mul(const enum CBLAS_TRANSPOSE TransA,
                const int M, const int N,
                const T alpha, const T *A, const T *X,
                const T beta, T *Y);
//template <typename T>
//void matvec_mul_gpu(const enum CBLAS_TRANSPOSE TransA,
//                    const int M, const int N,
//                    const T alpha, const T *A, const T *X,
//                    const T beta, T *Y);

template <typename T>
void MVMul(const enum CBLAS_TRANSPOSE TransA,
           const int M, const int N,
           const T alpha, const Tensor<CPU, T> &A, const Tensor<CPU, T> &X,
           const T beta, Tensor<CPU, T> &Y);

template <typename T>
void MVMul(const enum CBLAS_TRANSPOSE TransA,
           const int M, const int N,
           const T alpha, const Tensor<GPU, T> &A, const Tensor<GPU, T> &X,
           const T beta, Tensor<GPU, T> &Y);
/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */
// C = alpha * op(A)op(B) + beta * C
//template <typename T>
//void matrix_mul(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
//                const int M, const int N, const int K,
//                const T alpha, const T *A, const T *B,
//                const T beta, T *C);
//
//template <typename T>
//void matrix_mul_gpu(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
//                    const int M, const int N, const int K,
//                    const T alpha, const T *A, const T *B,
//                    const T beta, T *C);

template <typename T>
void MatMul(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
            const int M, const int N, const int K,
            const T alpha, const Tensor<CPU, T> &A, const Tensor<CPU, T> &B,
            const T beta, Tensor<CPU, T> &C);

template <typename T>
void MatMul(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
            const int M, const int N, const int K,
            const T alpha, const Tensor<GPU, T> &A, const Tensor<GPU, T> &B,
            const T beta, Tensor<GPU, T> &C);

/// 
template <typename T> void print_cpu(const int count, const T* A);
template <typename T> void print_gpu(const int count, const T* A);

//
template <typename T> void Copy(const Tensor<CPU, T>& X, Tensor<CPU, T>& Y);
template <typename T> void Copy(const Tensor<GPU, T>& X, Tensor<GPU, T>& Y);

template <typename T> T Dot(const Tensor<CPU, T>& X, Tensor<CPU, T>& Y);
template <typename T> T Dot(const Tensor<GPU, T>& X, Tensor<GPU, T>& Y);

template <typename T> void Scale(T alpha, Tensor<CPU, T>& X);
template <typename T> void Scale(T alpha, Tensor<GPU, T>& X);

template <typename T> void Add(const Tensor<CPU, T>& X1, const Tensor<CPU, T>& X2, Tensor<CPU, T>& Y);
template <typename T> void Add(const Tensor<GPU, T>& X1, const Tensor<GPU, T>& X2, Tensor<GPU, T>& Y);
template <typename T> void Sub(const Tensor<CPU, T>& X1, const Tensor<CPU, T>& X2, Tensor<CPU, T>& Y);
template <typename T> void Sub(const Tensor<GPU, T>& X1, const Tensor<GPU, T>& X2, Tensor<GPU, T>& Y);
template <typename T> void Mul(const Tensor<CPU, T>& X1, const Tensor<CPU, T>& X2, Tensor<CPU, T>& Y);
template <typename T> void Mul(const Tensor<GPU, T>& X1, const Tensor<GPU, T>& X2, Tensor<GPU, T>& Y);
template <typename T> void Div(const Tensor<CPU, T>& X1, const Tensor<CPU, T>& X2, Tensor<CPU, T>& Y);
template <typename T> void Div(const Tensor<GPU, T>& X1, const Tensor<GPU, T>& X2, Tensor<GPU, T>& Y);
template <typename T> void Exp(const Tensor<CPU, T>& X, Tensor<CPU, T>& Y);
template <typename T> void Exp(const Tensor<GPU, T>& X, Tensor<GPU, T>& Y);
template <typename T> void Sign(const Tensor<CPU, T>& X, Tensor<CPU, T>& Y);
template <typename T> void Sign(const Tensor<GPU, T>& X, Tensor<GPU, T>& Y);

template <typename T> void Sigmoid(const Tensor<CPU, T>& X, Tensor<CPU, T>& Y);
template <typename T> void Sigmoid(const Tensor<GPU, T>& X, Tensor<GPU, T>& Y);
template <typename T> void SigmoidGrad(const Tensor<CPU, T>& Y, const Tensor<CPU, T>& DY, Tensor<CPU, T>& DX);
template <typename T> void SigmoidGrad(const Tensor<GPU, T>& Y, const Tensor<GPU, T>& DY, Tensor<GPU, T>& DX);

template <typename T> void Tanh(const Tensor<CPU, T>& X, Tensor<CPU, T>& Y);
template <typename T> void Tanh(const Tensor<GPU, T>& X, Tensor<GPU, T>& Y);
template <typename T> void TanhGrad(const Tensor<CPU, T>& Y, const Tensor<CPU, T>& DY, Tensor<CPU, T>& DX);
template <typename T> void TanhGrad(const Tensor<GPU, T>& Y, const Tensor<GPU, T>& DY, Tensor<GPU, T>& DX);

template <typename T> void ReLU(const Tensor<CPU, T>& X, double alpha, Tensor<CPU, T>& Y);
template <typename T> void ReLU(const Tensor<GPU, T>& X, double alpha, Tensor<GPU, T>& Y);
template <typename T> void ReLUGrad(const Tensor<CPU, T>& X, const Tensor<CPU, T>& DY, double alpha, Tensor<CPU, T>& DX);
template <typename T> void ReLUGrad(const Tensor<GPU, T>& X, const Tensor<GPU, T>& DY, double alpha, Tensor<GPU, T>& DX);
}

#include "math_op_cpu.hpp"
#ifdef __CUDACC__
#include "math_op_gpu.hpp"
#endif
#endif //! ALCHEMY_MATH_MATH_OP_H
