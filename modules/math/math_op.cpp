#include "math_op.h"
#include <cstring>
#include <cstdint>
#include <cmath>
#include <iostream>

namespace alchemy {

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */
//template <>
//void vector_axpy(const int count, const float alpha, const float* X, float *Y)
//{
//    cblas_saxpy(count, alpha, X, 1, Y, 1);
//}

template <>
void axpy(float alpha, const Tensor<CPU, float> &X, Tensor<CPU, float> &Y)
{
    cblas_saxpy(X.size(), alpha, X.cptr(), 1, Y.mutable_cptr(), 1);
}

//template <>
//void vector_axpy(const int count, const double alpha, const double* X, double *Y)
//{
//    cblas_daxpy(count, alpha, X, 1, Y, 1);
//}

template <>
void axpy(double alpha, const Tensor<CPU, double> &X, Tensor<CPU, double> &Y)
{
    cblas_daxpy(X.size(), alpha, X.cptr(), 1, Y.mutable_cptr(), 1);
}

template <>
void Scale(const int count, const float alpha, float* X)
{
    cblas_sscal(count, alpha, X, 1);
}

template <>
void Scale(const int count, const double alpha, double* X)
{
    cblas_dscal(count, alpha, X, 1);
}

//template <>
//void vector_copy(const int count, const float* X, float* Y)
//{
//    cblas_scopy(count, X, 1, Y, 1);
//}
//
//template <>
//void vector_copy(const int count, const double* X, double* Y)
//{
//    cblas_dcopy(count, X, 1, Y, 1);
//}
template <>
void Scale(float alpha, Tensor<CPU, float>& X)
{
//    vector_scal(X.size(), alpha, X.mutable_cptr());
    cblas_sscal(X.size(), alpha, X.mutable_gptr(), 1);
//    cublasSscal(Global::cublas_handle(), X.size(), &alpha, X.mutable_gptr(), 1);
}

template <>
void Scale(double alpha, Tensor<CPU, double>& X)
{
    cblas_dscal(X.size(), alpha, X.mutable_gptr(), 1);
//    cublasDscal(Global::cublas_handle(), X.size(), &alpha, X.mutable_gptr(), 1);
}
template <>
void Copy(const Tensor<CPU, float>& X, Tensor<CPU, float>& Y)
{
    cblas_scopy(X.size(), X.cptr(), 1, Y.mutable_cptr(), 1);
}
template <>
void Copy(const Tensor<CPU, double>& X, Tensor<CPU, double>& Y)
{
    cblas_dcopy(X.size(), X.cptr(), 1, Y.mutable_cptr(), 1);
}

//template <>
//float vector_dot(const int count, const float* A, const float* B)
//{
//    return cblas_sdot(count, A, 1, B, 1);
//}
//
//template <>
//double vector_dot(const int count, const double* A, const double* B)
//{
//    return cblas_ddot(count, A, 1, B, 1);
//}

template <>
float Dot(const Tensor<CPU, float>& X, Tensor<CPU, float>& Y)
{
    return cblas_sdot(X.size(), X.cptr(), 1, Y.cptr(), 1);
}

template <>
double Dot(const Tensor<CPU, double>& X, Tensor<CPU, double>& Y)
{
    return cblas_ddot(X.size(), X.cptr(), 1, Y.cptr(), 1);
}

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */
template <>
void matvec_mul<CPU, float>(const enum CBLAS_TRANSPOSE TransA,
                       const int M, const int N,
                       const float alpha, const float *A, const float *X,
                       const float beta, float *Y)
{
    cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, X, 1, beta, Y, 1);
}

template <>
void matvec_mul<CPU, double>(const enum CBLAS_TRANSPOSE TransA,
                        const int M, const int N,
                        const double alpha, const double *A, const double *X,
                        const double beta, double *Y)
{
    cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, X, 1, beta, Y, 1);
}
template <>
void MVMul(const enum CBLAS_TRANSPOSE TransA,
           const int M, const int N,
           const float alpha, const Tensor<CPU, float> &A, const Tensor<CPU, float> &X,
           const float beta, Tensor<CPU, float> &Y)
{
    cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A.cptr(), N, X.cptr(), 1, beta, Y.mutable_cptr(), 1);
}

template <>
void MVMul(const enum CBLAS_TRANSPOSE TransA,
           const int M, const int N,
           const double alpha, const Tensor<CPU, double> &A, const Tensor<CPU, double> &X,
           const double beta, Tensor<CPU, double> &Y)
{
    cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A.cptr(), N, X.cptr(), 1, beta, Y.mutable_cptr(), 1);
}
/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */
//template <>
//void matrix_mul<float>(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
//                       const int M, const int N, const int K,
//                       const float alpha, const float *A, const float *B,
//                       const float beta,  float *C)
//{
//    auto lda = (TransA == CblasNoTrans) ? K : M;
//    auto ldb = (TransB == CblasNoTrans) ? N : K;
//    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
//}

template <>
void MatMul(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
            const int M, const int N, const int K,
            const float alpha, const Tensor<CPU, float> &A, const Tensor<CPU, float> &B,
            const float beta, Tensor<CPU, float> &C)
{
    auto lda = (TransA == CblasNoTrans) ? K : M;
    auto ldb = (TransB == CblasNoTrans) ? N : K;
    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A.cptr(), lda, B.cptr(), ldb, beta, C.mutable_cptr(), N);
}

//template <>
//void matrix_mul<double>(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
//                        const int M, const int N, const int K,
//                        const double alpha, const double *A, const double *B,
//                        const double beta,  double *C)
//{
//    auto lda = (TransA == CblasNoTrans) ? K : M;
//    auto ldb = (TransB == CblasNoTrans) ? N : K;
//    cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
//}
template <>
void MatMul(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
            const int M, const int N, const int K,
            const double alpha, const Tensor<CPU, double> &A, const Tensor<CPU, double> &B,
            const double beta, Tensor<CPU, double> &C)
{
    auto lda = (TransA == CblasNoTrans) ? K : M;
    auto ldb = (TransB == CblasNoTrans) ? N : K;
    cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A.cptr(), lda, B.cptr(), ldb, beta, C.mutable_cptr(), N);
}
///
template <typename T> void print_cpu(const int count, const T* A)
{
    std::cout << "CPU: ";
    for(auto i = 0; i < count; ++i) {
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;
}
template void print_cpu(const int count, const float* A);
template void print_cpu(const int count, const double* A);
}