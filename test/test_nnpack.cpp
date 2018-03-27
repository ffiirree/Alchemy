#include <alchemy.h>
#include <nnpack.h>
#include <iostream>

using namespace alchemy;
using namespace std;

int main()
{
    if(nnp_initialize() != nnp_status_success)
        return -1;

    Tensor<float> input({ 1, 1, 8, 8 });
    Tensor<float> output({ 1, 1, 6, 6 });
    Tensor<float> kernel({ 1, 1, 3, 3 });
    Tensor<float> bias({ 1, 1, 1, 1 });

    vector_set(input.count(), (float)2.0, input.mutable_cptr());
    vector_set(output.count(), (float)0.0, output.mutable_cptr());
    vector_set(kernel.count(), (float)1.0, kernel.mutable_cptr());
    vector_set(bias.count(), (float)0.3, bias.mutable_cptr());

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

    print_cpu(output.count(), output.cptr());

    return 0;
}