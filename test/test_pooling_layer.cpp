#include "zml/network.hpp"

using namespace z;

int main()
{
    auto kernel_size = 2;
    auto stride = 2;

    auto param = LayerParameter()
            .name("pooling layer")
            .type(POOLING_LAYER)
            .input("input_data")
            .output("output_data")
            .pooling_param(
                    PoolingParameter()
                            .kernel_size(kernel_size)
                            .stride(stride)
                            .type(MAX)
            );

    auto layer = LayerFactory<float>::GetLayer(param);

    vector<Tensor<float> *> input, output;
    input.push_back(new Tensor<float>({ 2, 3, 4, 6 }));

    output.push_back(new Tensor<float>());

    Filler<float>::fill(*input[0], NORMAL);

    for(auto i1 = 0; i1 <  input[0]->shape(0); ++i1) {
        for(auto i2 = 0; i2 < input[0]->shape(1); ++i2) {
            for(auto i3 = 0; i3 < input[0]->shape(2); ++i3) {
                for(auto i4 = 0; i4 < input[0]->shape(3); ++i4) {
                    std::cout << input[0]->data_at(i1, i2, i3, i4);
                    std::cout << ((i4 + 1 == input[0]->shape(3)) ? ";" : ", ");
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    layer->setup(input, output);
    layer->Forward(input, output);


    for(auto i1 = 0; i1 <  output[0]->shape(0); ++i1) {
        for(auto i2 = 0; i2 < output[0]->shape(1); ++i2) {
            for(auto i3 = 0; i3 < output[0]->shape(2); ++i3) {
                for(auto i4 = 0; i4 < output[0]->shape(3); ++i4) {
                    std::cout << output[0]->data_at(i1, i2, i3, i4);
                    std::cout << ((i4 + 1 == output[0]->shape(3)) ? ";" : ", ");
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }


    vector_set(output[0]->count(), (float)1.0, output[0]->diff());
    layer->Backward(input, output);

    for(auto i1 = 0; i1 <  input[0]->shape(0); ++i1) {
        for(auto i2 = 0; i2 < input[0]->shape(1); ++i2) {
            for(auto i3 = 0; i3 < input[0]->shape(2); ++i3) {
                for(auto i4 = 0; i4 < input[0]->shape(3); ++i4) {
                    std::cout << input[0]->diff_at(i1, i2, i3, i4);
                    std::cout << ((i4 + 1 == input[0]->shape(3)) ? ";" : ", ");
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }


    delete input[0];
    delete output[0];

    return 0;
}