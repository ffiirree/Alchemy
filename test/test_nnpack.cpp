#include <alchemy.h>
#include <nnpack.h>
#include <iostream>

using namespace alchemy;
using namespace std;

int main()
{
    if(nnp_initialize() != nnp_status_success)
        return -1;

    const size_t batch_size = 1;
    const size_t input_channels = 1;
    const size_t output_channels = 1;
    const struct nnp_padding input_padding{ 0, 0, 0, 0 };
    const struct nnp_size input_size = { 8, 8 };
    const struct nnp_size kernel_size = { 3, 3 };
    const struct nnp_size output_size = { 6, 6 };

    float* input =  (float *)malloc(batch_size * input_channels * input_size.width * input_size.height * sizeof(float));
    float* kernel = (float *)malloc(input_channels * output_channels * kernel_size.width * kernel_size.height * sizeof(float));
    float* output = (float *)malloc(batch_size * output_channels * output_size.height * output_size.width * sizeof(float));
    float* bias =   (float *)malloc(output_channels * sizeof(float));

    vector_set(64, (float)2.0, input);
    vector_set(36, (float)0.0, output);
    vector_set(9, (float)1.0, kernel);
    vector_set(1, (float)0.3, bias);


    nnp_convolution_output(nnp_convolution_algorithm_auto,
                           batch_size,
                           input_channels,
                           output_channels,
                           input_size,
                           input_padding,
                           kernel_size,
                           input,
                           kernel,
                           bias,
                           output,
                           nullptr,
                           nullptr);

    for(auto i = 0; i < 6; ++i) {
        for(auto j = 0; j < 6; ++j) {
            std::cout << output[i * 6 + j] << ", ";
        }
        std::cout << std::endl;
    }

    free(input);
    free(kernel);
    free(output);
    free(bias);

    return 0;
}