#include <cudnn_v7.h>
#include "alchemy.h"

using namespace std;
using namespace alchemy;

int main()
{
    // image
    auto image = imread("red.png");
    auto image_float = Matrix32f(image);

    //handle
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // input
    Tensor<float> input({ 1, image.channels(), image.rows_, image.cols_ });
    Memory::copy(image_float.count() * sizeof(float), input.mutable_gptr(), image_float.ptr());

    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor,
                               CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_FLOAT,
                               input.shape(0), input.shape(1), input.shape(2), input.shape(3));

    // output
    Tensor<float> output(input.shape());
    vector_set_gpu(output.count(), 0.0f, output.mutable_gptr());

    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(output_descriptor,
                               CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_FLOAT,
                               output.shape(0), output.shape(1), output.shape(2), output.shape(3));

    // kernel
    Tensor<float> kernel({ output.shape(1), input.shape(1), 3, 3 });
    auto kernel_size = kernel.count(2, 4);
    float kernel_[kernel_size] = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
    for(auto i = 0; i < kernel.count(0, 2); ++i) {
        memcpy(kernel.mutable_cptr() + i * kernel_size, kernel_, kernel_size * sizeof(float));
    }

    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnCreateFilterDescriptor(&kernel_descriptor);
    cudnnSetFilter4dDescriptor(kernel_descriptor,
                               CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NCHW,
                               kernel.shape(0), kernel.shape(1), kernel.shape(2), kernel.shape(3));
    // convolution descriptor
    cudnnConvolutionDescriptor_t conv_descriptor;
    cudnnCreateConvolutionDescriptor(&conv_descriptor);
    cudnnSetConvolution2dDescriptor(conv_descriptor,
                                    1, 1, // zero-padding
                                    1, 1, // stride
                                    1, 1,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // algorithm
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(handle,
                                        input_descriptor,
                                        kernel_descriptor,
                                        conv_descriptor,
                                        output_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        0,
                                        &algo);

    // workspace size && allocate memory
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle,
                                            input_descriptor,
                                            kernel_descriptor,
                                            conv_descriptor,
                                            output_descriptor,
                                            algo,
                                            &workspace_size);

    void * workspace = nullptr;
    cudaMalloc(&workspace, workspace_size);

    // convolution
    auto alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(handle,
                            &alpha, input_descriptor, input.gptr(),
                            kernel_descriptor, kernel.gptr(),
                            conv_descriptor, algo,
                            workspace, workspace_size,
                            &beta, output_descriptor, output.mutable_gptr());

    Matrix32f output_image(image.shape());
    cudaMemcpy(output_image.ptr(), output.gptr(), image.count() * sizeof(float), cudaMemcpyDeviceToHost);

    // destroy
    cudaFree(workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyConvolutionDescriptor(conv_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);

    cudnnDestroy(handle);


    // show
    imshow("original", image);
    imshow("output", Matrix(output_image/3.0));

    waitKey(0);
    return 0;
}
