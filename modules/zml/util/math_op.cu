#include <cstring>
#include <cstdint>
#include <zml/commen.hpp>
#include "math_op.hpp"
#include "cublas_v2.h"

namespace z {

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */
template <>
void vector_axpy(const int count, const float alpha, const float* X, float *Y)
{
    cublasSaxpy(Global::cublas_handle(), count, &alpha, X, 1, Y, 1);
}

template <>
void vector_axpy(const int count, const double alpha, const double* X, double *Y)
{
    cublasDaxpy(Global::cublas_handle(), count, &alpha, X, 1, Y, 1);
}

template <>
void vector_scal(const int count, const float alpha, float* X)
{
    cublasSscal(Global::cublas_handle(), count, &alpha, X, 1);
}

template <>
void vector_scal(const int count, const double alpha, double* X)
{
    cublasDscal(Global::cublas_handle(), count, &alpha, X, 1);
}

template <>
void vector_copy(const int count, const float* X, float* Y)
{
    cublasScopy(Global::cublas_handle(), count, X, 1, Y, 1);
}

template <>
void vector_copy(const int count, const double* X, double* Y)
{
    cublasDcopy(Global::cublas_handle(), count, X, 1, Y, 1);
}

//template <typename T>
//void vector_set(const int count, const T value, T* X)
//{
//    if(value == (T)0) {
//        std::memset(X, value, count * sizeof(T));
//        return ;
//    }
//
//    for(auto i = 0; i < count; ++i) {
//        X[i] = value;
//    }
//}
//template void vector_set<uint8_t>(const int count, const uint8_t value, uint8_t * X);
//template void vector_set<uint32_t>(const int count, const uint32_t value, uint32_t * X);
//template void vector_set<float>(const int count, const float value, float * X);
//template void vector_set<double>(const int count, const double value, double * X);
//
//template <typename T>
//void vector_sub(const int count, const T* A, const T* B, T* C)
//{
//    for(auto i = 0; i < count; ++i) {
//        C[i] = A[i] - B[i];
//    }
//}
//template void vector_sub<uint8_t>(const int count, const uint8_t* A, const uint8_t* B, uint8_t* C);
//template void vector_sub<uint32_t>(const int count, const uint32_t* A, const uint32_t* B, uint32_t* C);
//template void vector_sub<float>(const int count, const float* A, const float* B, float* C);
//template void vector_sub<double>(const int count, const double* A, const double* B, double* C);
//
//template <typename T>
//void vector_add(const int count, const T* A, const T* B, T* C)
//{
//    for(auto i = 0; i < count; ++i) {
//        C[i] = A[i] + B[i];
//    }
//}
//template void vector_add<uint8_t>(const int count, const uint8_t* A, const uint8_t* B, uint8_t* C);
//template void vector_add<uint32_t>(const int count, const uint32_t* A, const uint32_t* B, uint32_t* C);
//template void vector_add<float>(const int count, const float* A, const float* B, float* C);
//template void vector_add<double>(const int count, const double* A, const double* B, double* C);

template <>
float vector_dot(const int count, const float* A, const float* B)
{
    float result = 0;
    cublasSdot(Global::cublas_handle(), count, A, 1, B, 1, &result);
    return result;
}

template <>
double vector_dot(const int count, const double* A, const double* B)
{
    double result = 0;
    cublasDdot(Global::cublas_handle(), count, A, 1, B, 1, &result);
    return result;
}

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */
template <>
void matvec_mul<float>(const enum CBLAS_TRANSPOSE TransA,
                       const int M, const int N,
                       const float alpha, const float *A, const float *X,
                       const float beta, float *Y)
{
    cublasOperation_t transa = (TransA == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasSgemv(Global::cublas_handle(), transa, N, M, &alpha, A, N, X, 1, &beta, Y, 1);
}

template <>
void matvec_mul<double>(const enum CBLAS_TRANSPOSE TransA,
                        const int M, const int N,
                        const double alpha, const double *A, const double *X,
                        const double beta, double *Y)
{
    cublasOperation_t transa = (TransA == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasDgemv(Global::cublas_handle(), transa, N, M, &alpha, A, N, X, 1, &beta, Y, 1);
}

/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */
template <>
void matrix_mul<float>(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                       const int M, const int N, const int K,
                       const float alpha, const float *A, const float *B,
                       const float beta,  float *C)
{
    auto lda = (TransA == CblasNoTrans) ? K : M;
    auto ldb = (TransB == CblasNoTrans) ? N : K;
    cublasOperation_t transa = (TransA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = (TransB == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasSgemm(Global::cublas_handle(), transa, transb, N, M, K, &alpha, A, lda, B, ldb, &beta, C, N);
}


template <>
void matrix_mul<double>(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                        const int M, const int N, const int K,
                        const double alpha, const double *A, const double *B,
                        const double beta,  double *C)
{
    auto lda = (TransA == CblasNoTrans) ? K : M;
    auto ldb = (TransB == CblasNoTrans) ? N : K;
    cublasOperation_t transa = (TransA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = (TransB == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasDgemm(Global::cublas_handle(), transa, transb, N, M, K, &alpha, A, lda, B, ldb, &beta, C, N);
}


}