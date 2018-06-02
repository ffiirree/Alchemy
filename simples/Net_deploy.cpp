#include <alchemy.h>

using namespace alchemy;
using namespace std;


int main()
{
    MnistLoader<float> train_loader("/home/ffiirree/Code/Alchemy/resources/mnist/train-images.idx3-ubyte",
                                    "/home/ffiirree/Code/Alchemy/resources/mnist/train-labels.idx1-ubyte");


    vector<LayerParameter> params = {
            LayerParameter()
                    .name("mnist")
                    .type(INPUT_LAYER)
                    .output("data")
                    .output("label")
                    .input_param(
                            InputParameter()
                                    .source(&train_loader)
                                    .batch_size(64)
                                    .scale(1./255)
                    ),
            LayerParameter()
                    .name("cudnn_conv_01")
                    .type(CUDNN_CONV_LAYER)
                    .input("data")
                    .output("conv_01")
                    .conv_param(
                            ConvolutionParameter()
                                    .output_size(20)
                                    .kernel_size(5)
                                    .wlr(1)
                                    .blr(2)
                                    .weight_decay(0.0005)
                                    .weight_filler(XAVIER)
                                    .bias_filler(CONSTANT)
                    ),
            LayerParameter()
                    .name("pool_01")
                    .type(POOLING_LAYER)
                    .input("conv_01")
                    .output("pool_01")
                    .pooling_param(
                            PoolingParameter()
                                    .kernel_size(2)
                                    .stride(2)
                                    .type(MAX)
                    ),
            LayerParameter()
                    .name("cudnn_conv_02")
                    .type(CUDNN_CONV_LAYER)
                    .input("data")
                    .output("conv_02")
                    .conv_param(
                            ConvolutionParameter()
                                    .output_size(50)
                                    .kernel_size(5)
                                    .wlr(1)
                                    .blr(2)
                                    .weight_decay(0.0005)
                                    .weight_filler(XAVIER)
                                    .bias_filler(CONSTANT)
                    ),
            LayerParameter()
                    .name("pool_02")
                    .type(POOLING_LAYER)
                    .input("conv_02")
                    .output("pool_02")
                    .pooling_param(
                            PoolingParameter()
                                    .kernel_size(2)
                                    .stride(2)
                                    .type(MAX)
                    ),
            LayerParameter()
                    .name("ip_01")
                    .type(INNER_PRODUCT_LAYER)
                    .input("pool_02")
                    .output("ip_01")
                    .ip_param(
                            InnerProductParameter()
                                    .output_size(500)
                                    .wlr(0.2)
                                    .blr(0.4)
                                    .weight_decay(0.0005)
                                    .weight_filler(XAVIER)
                                    .bias_filler(CONSTANT)
                    ),
            LayerParameter()
                    .name("relu_01")
                    .type(RELU_LAYER)
                    .input("ip_01")
                    .output("act_01")
                    .relu_param(
                            ReLuParameter()
                                    .alpha(-0.2)
                    ),
            LayerParameter()
                    .name("ip_02")
                    .type(INNER_PRODUCT_LAYER)
                    .input("act_01")
                    .output("ip_02")
                    .ip_param(
                            InnerProductParameter()
                                    .output_size(10)
                                    .wlr(0.2)
                                    .blr(0.4)
                                    .weight_decay(0.0005)
                                    .weight_filler(XAVIER)
                                    .bias_filler(CONSTANT)
                    ),
            LayerParameter()
                    .name("accuracy")
                    .type(ACCURACY_LAYER)
                    .input("ip_02")
                    .input("label")
                    .output("accuracy")
                    .accuracy_param(
                            AccuracyParameter()
                    )
    };

    Network<GPU, float> net(NetworkParameter().layer_params(params));
    net.load("LeNet.alc");
    net.Forward();
    print_cpu(net.inputs().back()[0]->size(), net.inputs().back()[0]->data_cptr());
    std::cout << net.accuracy();

    return 0;
}
