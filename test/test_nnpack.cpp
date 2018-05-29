#include <alchemy.h>
#include <nnpack.h>
#include <iostream>

using namespace alchemy;
using namespace std;

int main()
{
    if(nnp_initialize() != nnp_status_success)
        return -1;

    Tensor<CPU, float> input({ 1, 1, 8, 8 });
    Tensor<CPU, float> output({ 1, 1, 6, 6 });
    Tensor<CPU, float> kernel({ 1, 1, 3, 3 });
    Tensor<CPU, float> bias({ 1, 1, 1, 1 });

    vector_set(input.size(), (float)2.0, input.mutable_cptr());
    vector_set(output.size(), (float)0.0, output.mutable_cptr());
    vector_set(kernel.size(), (float)1.0, kernel.mutable_cptr());
    vector_set(bias.size(), (float)0.3, bias.mutable_cptr());

    const struct nnp_padding input_padding{ 0, 0, 0, 0 };
    const struct nnp_size input_size = {static_cast<size_t>(input.shape(2)), static_cast<size_t>(input.shape(3))};
    const struct nnp_size kernel_size = {static_cast<size_t>(kernel.shape(2)), static_cast<size_t>(kernel.shape(3))};

    nnp_convolution_output(nnp_convolution_algorithm_auto,
                           static_cast<size_t>(input.shape(0)),
                           static_cast<size_t>(input.shape(1)),
                           static_cast<size_t>(output.shape(1)),
                           input_size,
                           input_padding,
                           kernel_size,
                           input.cptr(),
                           kernel.cptr(),
                           bias.cptr(),
                           output.mutable_cptr(),
                           nullptr,
                           nullptr);

    print_cpu(output.size(), output.cptr());

    return 0;
}