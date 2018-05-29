#include <alchemy.h>

using namespace alchemy;
using namespace std;


int main()
{
    SandLoader<float> train_loader("/home/ffiirree/Code/Alchemy/resources/train_images_20k.ubyte",
                                   "/home/ffiirree/Code/Alchemy/resources/train_labels_20k.ubyte");

    SandLoader<float> test_loader("/home/ffiirree/Code/Alchemy/resources/test_images_10k.ubyte",
                                  "/home/ffiirree/Code/Alchemy/resources/test_labels_10k.ubyte");

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
                                    .batch_size(1)
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
                                    .batch_size(100)
                                    .scale(1./255)
                    ),
            LayerParameter()
                    .name("conv_01")
                    .type(CUDNN_CONV_LAYER)
                    .input("data")
                    .output("conv_01")
                    .conv_param(
                            ConvolutionParameter()
                                    .output_size(20)
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
                                    .output_size(50)
                                    .wlr(0.5)
                                    .blr(1)
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
            LayerParameter()
                    .name("ip_02")
                    .type(INNER_PRODUCT_LAYER)
                    .input("sigmoid_01")
                    .output("ip_02")
                    .ip_param(
                            InnerProductParameter()
                                    .output_size(1)
                                    .wlr(0.5)
                                    .blr(1)
                                    .weight_filler(XAVIER)
                                    .bias_filler(CONSTANT)
                    ),

            LayerParameter()
                    .name("sigmoid_02")
                    .type(SIGMOID_LAYER)
                    .input("ip_02")
                    .output("sigmoid_02")
                    .sigmoid_param(
                            SigmoidParameter()
                    ),
            LayerParameter()
                    .name("loss")
                    .type(EUCLIDEAN_LOSS_LAYER)
                    .phase(TRAIN)
                    .input("sigmoid_02")
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
            .mode(Global::GPU)
            .max_iter(10000)
            .test_iter(1)
            .test_interval(1)
            .train_net_param(NetworkParameter().layer_params(params))
            .test_net_param(NetworkParameter().layer_params(params));

    SgdOptimizer<GPU, float> o(optimize_param);


//    o.load("/home/ffiirree/Code/Alchemy/resources/train_.bin");
    o.optimize();
//    o.save("/home/ffiirree/Code/Alchemy/resources/train_.bin");

    return 0;
}
