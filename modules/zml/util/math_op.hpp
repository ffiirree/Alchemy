#ifndef _ZML_MATH_OP_HPP
#define _ZML_MATH_OP_HPP

extern "C"{
#include "cblas.h"
};


namespace z {

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

// Y = alpha * X + Y
template <typename T> void vector_axpy(const int count, const T alpha, const T* X, T *Y);
// X = alpha * X
template <typename T> void vector_scal(const int count, const T alpha, T* X);
// Y = X
template <typename T> void vector_copy(const int count, const T* X, T* Y);
// foreach Xi = value
template <typename T> void vector_set(const int count, const T value, T* X);
// C = A - B
template <typename T> void vector_sub(const int count, const T* A, const T* B, T* C);
// C = A + B
template <typename T> void vector_add(const int count, const T* A, const T* B, T* C);
// return t(A)B
template <typename T> T    vector_dot(const int count, const T* A, const T* B);
// B = exp(A)
template <typename T> void vector_exp(const int count, const T* A, T* B);
//
template <typename T> void vector_div(const int count, const T* A, const T* B, T* C);
//
template <typename T> void vector_sign(const int count, const T* A, T* B);

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



}
#endif //_ZML_MATH_OP_HPP
