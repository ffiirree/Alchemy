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
                    .name("mnist")
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
                    .name("mnist")
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
                    .name("conv_01")
                    .type(CUDNN_CONV_LAYER)
                    .input("data")
                    .output("conv_01")
                    .conv_param(
                            ConvolutionParameter()
                                    .output_size(10)
                                    .kernel_size(5)
                                    .wlr(1)
                                    .blr(2)
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
                    .name("ip_01")
                    .type(INNER_PRODUCT_LAYER)
                    .input("pool_01")
                    .output("ip_01")
                    .ip_param(
                            InnerProductParameter()
                                    .output_size(30)
                                    .wlr(0.01)
                                    .blr(0.02)
                                    .weight_filler(XAVIER)
                                    .bias_filler(CONSTANT)
                    ),
            LayerParameter()
                    .name("sigmoid_01")
                    .type(SIGMOID_LAYER)
                    .input("ip_01")
                    .output("sigmoid_01")
                    .sigmoid_param(
                            SigmoidParameter()
                    ),

//            LayerParameter()
//                    .name("relu_01")
//                    .type(RELU_LAYER)
//                    .input("ip_01")
//                    .output("sigmoid_01")
//                    .relu_param(
//                            ReLuParameter().alpha(-0.1)
//                    ),
            LayerParameter()
                    .name("ip_02")
                    .type(INNER_PRODUCT_LAYER)
                    .input("sigmoid_01")
                    .output("ip_02")
                    .ip_param(
                            InnerProductParameter()
                                    .output_size(10)
                                    .wlr(0.01)
                                    .blr(0.02)
                                    .weight_filler(XAVIER)
                                    .bias_filler(CONSTANT)
                    ),
            LayerParameter()
                    .name("loss")
                    .type(SOFTMAX_LOSS_LAYER)
                    .phase(TRAIN)
                    .input("ip_02")
                    .input("label")
                    .output("loss")
                    .softmax_loss_param(
                            SoftmaxLossParameter()
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
            .mode(Global::GPU)
            .max_iter(60000)
            .test_iter(100)
            .test_interval(500)
            .train_net_param(NetworkParameter().layer_params(params))
            .test_net_param(NetworkParameter().layer_params(params));

    SgdOptimizer<GPU, float> o(optimize_param);

    o.optimize();

    return 0;
}
