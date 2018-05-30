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
                                    .batch_size(64)
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
                                    .batch_size(64)
                                    .scale(1./255)
                    ),
            LayerParameter()
                    .name("ip_01")
                    .type(INNER_PRODUCT_LAYER)
                    .input("data")
                    .output("ip_01")
                    .ip_param(
                            InnerProductParameter()
                                    .output_size(50)
                                    .wlr(0.1)
                                    .blr(0.2)
                                    .weight_filler(XAVIER)
                                    .bias_filler(CONSTANT)
                    ),
            LayerParameter()
                    .name("sig_01")
                    .type(SIGMOID_LAYER)
                    .input("ip_01")
                    .output("act_01")
                    .sigmoid_param(
                            SigmoidParameter()
                    ),
            LayerParameter()
                    .name("ip_02")
                    .type(INNER_PRODUCT_LAYER)
                    .input("act_01")
                    .output("ip_02")
                    .ip_param(
                            InnerProductParameter()
                                    .output_size(10)
                                    .wlr(0.1)
                                    .blr(0.2)
                                    .weight_filler(XAVIER)
                                    .bias_filler(CONSTANT)
                    ),
            LayerParameter()
                    .name("loss")
                    .phase(TRAIN)
                    .type(SOFTMAX_LOSS_LAYER)
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
            .max_iter(20000)
            .test_iter(100)
            .test_interval(200)
            .train_net_param(NetworkParameter().layer_params(params))
            .test_net_param(NetworkParameter().layer_params(params));

    SgdOptimizer<GPU, float> o(optimize_param);

    o.optimize();

    return 0;
}
