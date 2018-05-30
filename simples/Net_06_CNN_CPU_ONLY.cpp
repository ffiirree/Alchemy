#include <alchemy.h>

using namespace alchemy;
using namespace std;

int main()
{
    MnistLoader<float> train_loader("/home/ffiirree/Code/Alchemy/resources/mnist/train-images.idx3-ubyte",
                                    "/home/ffiirree/Code/Alchemy/resources/mnist/train-labels.idx1-ubyte");

    MnistLoader<float> test_loader("/home/ffiirree/Code/Alchemy/resources/mnist/t10k-images.idx3-ubyte",
                                   "/home/ffiirree/Code/Alchemy/resources/mnist/t10k-labels.idx1-ubyte");

    vector<LayerParameter> params = {
            LayerParameter()
                    .name("mnist_train")
                    .type(INPUT_LAYER)
                    .phase(TRAIN)
                    .output("data")
                    .output("label")
                    .input_param(
                            InputParameter()
                                    .source(&train_loader)
                                    .batch_size(10)
                                    .scale(1./255)
                    ),
            LayerParameter()
                    .name("mnist_test")
                    .type(INPUT_LAYER)
                    .phase(TEST)
                    .output("data")
                    .output("label")
                    .input_param(
                            InputParameter()
                                    .source(&test_loader)
                                    .batch_size(10)
                                    .scale(1./255)
                    ),
            LayerParameter()
                    .name("conv_layer_01")
                    .type(CONVOLUTION_LAYER)
                    .input("data")
                    .output("conv_01")
                    .conv_param(
                            ConvolutionParameter()
                                    .output_size(20)
                                    .kernel_size(5)
                                    .wlr(1.5)
                                    .blr(3)
                                    .weight_decay(0.05)
                                    .weight_filler(XAVIER)
                                    .bias_filler(XAVIER)
                    ),
            LayerParameter()
                    .name("pooling_layer_01")
                    .type(POOLING_LAYER)
                    .input("conv_01")
                    .output("pooling_01")
                    .pooling_param(
                            PoolingParameter()
                                    .kernel_size(2)
                                    .stride(2)
                                    .type(MAX)
                    ),
            LayerParameter()
                    .name("ip_layer_01")
                    .type(INNER_PRODUCT_LAYER)
                    .input("pooling_01")
                    .output("ip_01")
                    .ip_param(
                            InnerProductParameter()
                                    .output_size(100)
                                    .wlr(0.15)
                                    .blr(0.3)
                                    .weight_decay(0.05)
                                    .weight_filler(XAVIER)
                                    .bias_filler(XAVIER)
                    ),
            LayerParameter()
                    .name("sigmoid_layer_01")
                    .type(SIGMOID_LAYER)
                    .input("ip_01")
                    .output("sigmoid_01")
                    .sigmoid_param(
                            SigmoidParameter()
                    ),
            LayerParameter()
                    .name("ip_layer_02")
                    .type(INNER_PRODUCT_LAYER)
                    .input("sigmoid_01")
                    .output("ip_02")
                    .ip_param(
                            InnerProductParameter()
                                    .output_size(10)
                                    .wlr(0.15)
                                    .blr(0.3)
                                    .weight_filler(XAVIER)
                                    .bias_filler(XAVIER)
                    ),
            LayerParameter()
                    .name("loss")
                    .type(SIGMOID_CROSS_ENTORPY_LOSS_LAYER)
                    .phase(TRAIN)
                    .input("ip_02")
                    .input("label")
                    .output("loss")
                    .euclidean_param(
                            EuclideanLossParameter()
                    ),
            LayerParameter()
                    .name("accuracy")
                    .type(ACCURACY_LAYER)
                    .phase(TEST)
                    .input("ip_02")
                    .input("label")
                    .output("accuracy")
                    .accuracy_param(
                            AccuracyParameter()
                    )
    };

    auto optimize_param = OptimizerParameter()
            .mode(Global::CPU)
            .max_iter(60000)
            .test_iter(100)
            .test_interval(500)
            .train_net_param(NetworkParameter().layer_params(params))
            .test_net_param(NetworkParameter().layer_params(params));

    SgdOptimizer<CPU, float> o(optimize_param);

    o.optimize();

    return 0;
}
