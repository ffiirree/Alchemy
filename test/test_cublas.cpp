#include <iostream>
#include <alchemy.h>

using namespace alchemy;
using namespace std;

int main()
{

    /// gemm
//    Tensor<float> t_1({4, 2});
//    Tensor<float> t_2({3, 2});
//    Tensor<float> t_3({4, 3});
//
//    auto t_1_ptr = t_1.cpu_data();
//    auto t_2_ptr = t_2.cpu_data();
//    t_1_ptr[0] = 1.5f;
//    t_1_ptr[1] = 3.2f;
//    t_1_ptr[2] = 3.0f;
//    t_1_ptr[3] = 12.0f;
//    t_1_ptr[4] = -2.0f;
//    t_1_ptr[5] = 4.0f;
//    t_1_ptr[6] = 2.3f;
//    t_1_ptr[7] = 4.2f;
//
//    t_2_ptr[0] = 2.0f;
//    t_2_ptr[1] = 1.0f;
//    t_2_ptr[2] = 3.0f;
//    t_2_ptr[3] = 2.0f;
//    t_2_ptr[4] = 1.0f;
//    t_2_ptr[5] = 4.0f;
//
//
//    int M = 4;
//    int N = 3;
//    int K = 2;
//    matrix_mul(CblasNoTrans, CblasTrans, M, N, K, (float)1., t_1.cpu_data(), t_2.cpu_data(), (float)0., t_3.cpu_data());
//    print_cpu(t_3.count(), t_3.cpu_data());
//
//    float alpha = 1.0f;
//    float beta = 0.0f;
//    auto status = cublasSgemm(Global::cublas_handle(),
//                              CUBLAS_OP_T, CUBLAS_OP_N,
//                              N, M, K,
//                              &alpha, t_2.gpu_data(), K, t_1.gpu_data(), K,
//                              &beta, t_3.gpu_data(), N);
//
//    print_gpu(t_3.count(), t_3.gpu_data());


    ///
    Tensor<float> axpy1({2, 2});
    Tensor<float> axpy2({2, 2});

    Filler<float>::fill(axpy1, NORMAL);
    Filler<float>::fill(axpy2, XAVIER);

    cout << "[before/0]"; print_cpu(axpy1.count(), axpy1.cpu_data());
    cout << "[before/1]"; print_cpu(axpy2.count(), axpy2.cpu_data());
    vector_axpy_gpu(axpy1.count(), -1.f, axpy1.gpu_data(), axpy2.gpu_data());
    cout << "[after /1]"; print_cpu(axpy2.count(), axpy2.cpu_data());
    cout << "[after /1]"; print_gpu(axpy2.count(), axpy2.gpu_data());
    return 0;
}