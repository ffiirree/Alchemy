#include <alchemy.h>

using namespace alchemy;

int main()
{
    Global::mode(Global::GPU);

    Blob<float> input({2, 3, 6, 5});
    Blob<float> output;

    Filler<float>::uniform_fill(input.count(), input.mutable_data_cptr(), 0, 10.0);

    print_cpu(input.count(), input.data_cptr());

    PoolingLayer<float> poolingLayer(
            LayerParameter()
                    .pooling_param(
                            PoolingParameter()
                                    .kernel_size(2)
                                    .type(MAX)
                                    .stride(2)
                    )
    );

    poolingLayer.setup({ &input }, { &output });

    std::cout << "#GPU: \n";
    Global::mode(Global::GPU);
    poolingLayer.Forward({ &input }, { &output });
//    print_gpu(output.count(), output.data_gptr());
    print_cpu(output.count(), output.data_cptr());

    std::cout << "\n#CPU: \n";
    Global::mode(Global::CPU);
    poolingLayer.Forward({ &input }, { &output });
//    print_gpu(output.count(), output.data_gptr());
    print_cpu(output.count(), output.data_cptr());

    return 0;
}
