#include "math_op.h"
#include <cstring>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include "core/common.h"

namespace alchemy {

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */
//template <>
//void vector_axpy_gpu(const int count, const float alpha, const float* X, float *Y)
//{
//    cublasSaxpy(Global::cublas_handle(), count, &alpha, X, 1, Y, 1);
//}
template <>
void axpy(float alpha, const Tensor<GPU, float> &X, Tensor<GPU, float> &Y)
{
    cublasSaxpy(Global::cublas_handle(), X.size(), &alpha, X.ptr(), 1, Y.mutable_ptr(), 1);
}

//template <>
//void vector_axpy_gpu(const int count, const double alpha, const double* X, double *Y)
//{
//    cublasDaxpy(Global::cublas_handle(), count, &alpha, X, 1, Y, 1);
//}
template <>
void axpy(double alpha, const Tensor<GPU, double> &X, Tensor<GPU, double> &Y)
{
    cublasDaxpy(Global::cublas_handle(), X.size(), &alpha, X.ptr(), 1, Y.mutable_ptr(), 1);
}
//template <>
//void vector_scal_gpu(const int count, const float alpha, float* X)
//{
//    cublasSscal(Global::cublas_handle(), count, &alpha, X, 1);
//}
//
//template <>
//void vector_scal_gpu(const int count, const double alpha, double* X)
//{
//    cublasDscal(Global::cublas_handle(), count, &alpha, X, 1);
//}
//
//template <>
//void vector_copy_gpu(const int count, const float* X, float* Y)
//{
//    cublasScopy(Global::cublas_handle(), count, X, 1, Y, 1);
//}
//
//template <>
//void vector_copy_gpu(const int count, const double* X, double* Y)
//{
//    cublasDcopy(Global::cublas_handle(), count, X, 1, Y, 1);
//}
template <>
void Scale(float alpha, Tensor<GPU, float>& X)
{
//    vector_scal(X.size(), alpha, X.mutable_cptr());
    cublasSscal(Global::cublas_handle(), X.size(), &alpha, X.mutable_ptr(), 1);
}

template <>
void Scale(double alpha, Tensor<GPU, double>& X)
{
    cublasDscal(Global::cublas_handle(), X.size(), &alpha, X.mutable_ptr(), 1);
}

template <>
void Copy(const Tensor<GPU, float>& X, Tensor<GPU, float>& Y)
{
    cublasScopy(Global::cublas_handle(), X.size(), X.ptr(), 1, Y.mutable_ptr(), 1);
}
template <>
void Copy(const Tensor<GPU, double>& X, Tensor<GPU, double>& Y)
{
    cublasDcopy(Global::cublas_handle(), X.size(), X.ptr(), 1, Y.mutable_ptr(), 1);
}
//
//template <>
//float vector_dot_gpu(const int count, const float* A, const float* B)
//{
//    float result = 0;
//    cublasSdot(Global::cublas_handle(), count, A, 1, B, 1, &result);
//    return result;
//}
//
//template <>
//double vector_dot_gpu(const int count, const double* A, const double* B)
//{
//    double result = 0;
//    cublasDdot(Global::cublas_handle(), count, A, 1, B, 1, &result);
//    return result;
//}

template <>
float Dot(const Tensor<GPU, float>& X, Tensor<GPU, float>& Y)
{
    float result = 0;
    cublasSdot(Global::cublas_handle(), X.size(), X.ptr(), 1, Y.ptr(), 1, &result);
    return result;
}

template <>
double Dot(const Tensor<GPU, double>& X, Tensor<GPU, double>& Y)
{
    double result = 0;
    cublasDdot(Global::cublas_handle(), X.size(), X.ptr(), 1, Y.ptr(), 1, &result);
    return result;
}
/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */
template <>
void matvec_mul<GPU, float>(const enum CBLAS_TRANSPOSE TransA,
                           const int M, const int N,
                           const float alpha, const float *A, const float *X,
                           const float beta, float *Y)
{
    cublasOperation_t transa = (TransA == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasSgemv(Global::cublas_handle(), transa, N, M, &alpha, A, N, X, 1, &beta, Y, 1);
}

template <>
void matvec_mul<GPU, double>(const enum CBLAS_TRANSPOSE TransA,
                            const int M, const int N,
                            const double alpha, const double *A, const double *X,
                            const double beta, double *Y)
{
    cublasOperation_t transa = (TransA == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasDgemv(Global::cublas_handle(), transa, N, M, &alpha, A, N, X, 1, &beta, Y, 1);
}

template <>
void MVMul(const enum CBLAS_TRANSPOSE TransA,
           const int M, const int N,
           const float alpha, const Tensor<GPU, float> &A, const Tensor<GPU, float> &X,
           const float beta, Tensor<GPU, float> &Y)
{
    cublasOperation_t transa = (TransA == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasSgemv(Global::cublas_handle(), transa, N, M, &alpha, A.ptr(), N, X.ptr(), 1, &beta, Y.mutable_ptr(), 1);
}

template <>
void MVMul(const enum CBLAS_TRANSPOSE TransA,
           const int M, const int N,
           const double alpha, const Tensor<GPU, double> &A, const Tensor<GPU, double> &X,
           const double beta, Tensor<GPU, double> &Y)
{
    cublasOperation_t transa = (TransA == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasDgemv(Global::cublas_handle(), transa, N, M, &alpha, A.ptr(), N, X.ptr(), 1, &beta, Y.mutable_ptr(), 1);
}

/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */
//template <>
//void matrix_mul_gpu<float>(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
//                           const int M, const int N, const int K,
//                           const float alpha, const float *A, const float *B,
//                           const float beta,  float *C)
//{
//    auto lda = (TransA == CblasNoTrans) ? K : M;
//    auto ldb = (TransB == CblasNoTrans) ? N : K;
//    cublasOperation_t transa = (TransA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
//    cublasOperation_t transb = (TransB == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
//
//    cublasSgemm(Global::cublas_handle(), transb, transa, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);
//}

template <>
void MatMul(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
            const int M, const int N, const int K,
            const double alpha, const Tensor<GPU, double> &A, const Tensor<GPU, double> &B,
            const double beta, Tensor<GPU, double> &C)
{
    auto lda = (TransA == CblasNoTrans) ? K : M;
    auto ldb = (TransB == CblasNoTrans) ? N : K;
    cublasOperation_t transa = (TransA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = (TransB == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasDgemm(Global::cublas_handle(), transb, transa, N, M, K, &alpha, B.ptr(), ldb, A.ptr(), lda, &beta, C.mutable_ptr(), N);
}

//template <>
//void matrix_mul_gpu<double>(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
//                            const int M, const int N, const int K,
//                            const double alpha, const double *A, const double *B,
//                            const double beta,  double *C)
//{
//    auto lda = (TransA == CblasNoTrans) ? K : M;
//    auto ldb = (TransB == CblasNoTrans) ? N : K;
//    cublasOperation_t transa = (TransA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
//    cublasOperation_t transb = (TransB == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
//
//    cublasDgemm(Global::cublas_handle(), transb, transa, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);
//}

template <>
void MatMul(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
            const int M, const int N, const int K,
            const float alpha, const Tensor<GPU, float> &A, const Tensor<GPU, float> &B,
            const float beta, Tensor<GPU, float> &C)
{
    auto lda = (TransA == CblasNoTrans) ? K : M;
    auto ldb = (TransB == CblasNoTrans) ? N : K;
    cublasOperation_t transa = (TransA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = (TransB == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasSgemm(Global::cublas_handle(), transb, transa, N, M, K, &alpha, B.ptr(), ldb, A.ptr(), lda, &beta, C.mutable_ptr(), N);
}

///
__global__ void print_kernel(const int count, const float* A) {
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        if(i == 0) printf("GPU: ");
        printf("(%d, %f), ", i, A[i]);
        if(i + 1 == count) printf("\n");
    }
}
template <>
void print_gpu(const int count, const float* A)
{
    print_kernel<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, A);
}

__global__ void print_kernel(const int count, const double* A) {
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        if(i == 0) printf("GPU: ");
        printf("(%d, %f), ", i, A[i]);
        if(i + 1 == count) printf("\n");
    }
}
template <>
void print_gpu(const int count, const double* A)
{
    print_kernel<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, A);
}
}