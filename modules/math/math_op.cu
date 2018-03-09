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
template <>
void vector_axpy_gpu(const int count, const float alpha, const float* X, float *Y)
{
    cublasSaxpy(Global::cublas_handle(), count, &alpha, X, 1, Y, 1);
}

template <>
void vector_axpy_gpu(const int count, const double alpha, const double* X, double *Y)
{
    cublasDaxpy(Global::cublas_handle(), count, &alpha, X, 1, Y, 1);
}

template <>
void vector_scal_gpu(const int count, const float alpha, float* X)
{
    cublasSscal(Global::cublas_handle(), count, &alpha, X, 1);
}

template <>
void vector_scal_gpu(const int count, const double alpha, double* X)
{
    cublasDscal(Global::cublas_handle(), count, &alpha, X, 1);
}

template <>
void vector_copy_gpu(const int count, const float* X, float* Y)
{
    cublasScopy(Global::cublas_handle(), count, X, 1, Y, 1);
}

template <>
void vector_copy_gpu(const int count, const double* X, double* Y)
{
    cublasDcopy(Global::cublas_handle(), count, X, 1, Y, 1);
}

template <typename T>
__global__ void set_kernel(const int count, const T value, T* X) {
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        X[i] = value;
    }
}

template <typename T>
void vector_set_gpu(const int count, const T value, T* X)
{
    if(value == (T)0) {
        cudaMemset(X, value, count * sizeof(T));
        return ;
    }

    set_kernel<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, value, X);
}
template void vector_set_gpu<float>(const int count, const float value, float * X);
template void vector_set_gpu<double>(const int count, const double value, double * X);

template <typename T>
__global__ void sub_kernel(const int count, const T* A, const T* B,T* C) {
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        C[i] = A[i] - B[i];
    }
}

template <typename T>
void vector_sub_gpu(const int count, const T* A, const T* B, T* C)
{
    sub_kernel<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, A, B, C);
}
template void vector_sub_gpu<float>(const int count, const float* A, const float* B, float* C);
template void vector_sub_gpu<double>(const int count, const double* A, const double* B, double* C);

template <typename T>
__global__ void add_kernel(const int count, const T* A, const T* B,T* C) {
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        C[i] = A[i] + B[i];
    }
}

template <typename T>
void vector_add_gpu(const int count, const T* A, const T* B, T* C)
{
    add_kernel<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, A, B, C);
}
template void vector_add_gpu<float>(const int count, const float* A, const float* B, float* C);
template void vector_add_gpu<double>(const int count, const double* A, const double* B, double* C);

template <>
float vector_dot_gpu(const int count, const float* A, const float* B)
{
    float result = 0;
    cublasSdot(Global::cublas_handle(), count, A, 1, B, 1, &result);
    return result;
}

template <>
double vector_dot_gpu(const int count, const double* A, const double* B)
{
    double result = 0;
    cublasDdot(Global::cublas_handle(), count, A, 1, B, 1, &result);
    return result;
}

template <typename T>
__global__ void exp_kernel(const int count, const T* A, T* B) {
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        B[i] = std::exp(A[i]);
    }
}

template <typename T>
void vector_exp_gpu(const int count, const T* A, T* B)
{
    exp_kernel<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, A, B);
}
template void vector_exp_gpu(const int count, const float* A, float* B);
template void vector_exp_gpu(const int count, const double* A, double* B);

template <typename T>
__global__ void div_kernel(const int count, const T* A, const T* B, T* C) {
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        C[i] = A[i] / B[i];
    }
}

template <typename T>
void vector_div_gpu(const int count, const T* A, const T* B, T* C)
{
    div_kernel<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, A, B, C);
}
template void vector_div_gpu(const int count, const float* A, const float* B, float* C);
template void vector_div_gpu(const int count, const double* A, const double* B, double* C);

template <typename T>
__global__ void sign_kernel(const int count, const T* A, T* B) {
    for(auto i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        B[i] = (A[i] > 0) - (0 > A[i]);
    }
}

template <typename T>
void vector_sign_gpu(const int count, const T* A, T*B)
{
    sign_kernel<<<CUDA_BLOCK_NUM(count), CUDA_THREAD_NUM>>>(count, A, B);
}
template void vector_sign_gpu(const int count, const float* A, float* B);
template void vector_sign_gpu(const int count, const double* A, double* B);

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */
template <>
void matvec_mul_gpu<float>(const enum CBLAS_TRANSPOSE TransA,
                           const int M, const int N,
                           const float alpha, const float *A, const float *X,
                           const float beta, float *Y)
{
    cublasOperation_t transa = (TransA == CblasTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasSgemv(Global::cublas_handle(), transa, N, M, &alpha, A, N, X, 1, &beta, Y, 1);
}

template <>
void matvec_mul_gpu<double>(const enum CBLAS_TRANSPOSE TransA,
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
void matrix_mul_gpu<float>(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                           const int M, const int N, const int K,
                           const float alpha, const float *A, const float *B,
                           const float beta,  float *C)
{
    auto lda = (TransA == CblasNoTrans) ? K : M;
    auto ldb = (TransB == CblasNoTrans) ? N : K;
    cublasOperation_t transa = (TransA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = (TransB == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasSgemm(Global::cublas_handle(), transb, transa, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);
}


template <>
void matrix_mul_gpu<double>(const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
                            const int M, const int N, const int K,
                            const double alpha, const double *A, const double *B,
                            const double beta,  double *C)
{
    auto lda = (TransA == CblasNoTrans) ? K : M;
    auto ldb = (TransB == CblasNoTrans) ? N : K;
    cublasOperation_t transa = (TransA == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = (TransB == CblasTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasDgemm(Global::cublas_handle(), transb, transa, N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);
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