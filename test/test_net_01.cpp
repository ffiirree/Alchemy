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
                                    .batch_size(100)
                                    .scale(1./255)
                    ),
            LayerParameter()
                    .name("inner_product_01")
                    .type(INNER_PRODUCT_LAYER)
                    .input("data")
                    .output("ip1")
                    .ip_param(
                            InnerProductParameter()
                                    .output_size(30)
                                    .wlr(0.3)
                                    .blr(0.3)
                                    .weight_decay(0.005)
                                    .weight_filler(XAVIER)
                                    .bias_filler(XAVIER)
                    ),
            LayerParameter()
                    .name("sigmoid_layer_01")
                    .type(SIGMOID_LAYER)
                    .input("ip1")
                    .output("s1")
                    .sigmoid_param(
                            SigmoidParameter()
                    ),
            LayerParameter()
                    .name("inner_product_02")
                    .type(INNER_PRODUCT_LAYER)
                    .input("s1")
                    .output("ip2")
                    .ip_param(
                            InnerProductParameter()
                                    .output_size(10)
                                    .wlr(0.3)
                                    .blr(0.3)
                                    .weight_decay(0.005)
                                    .weight_filler(XAVIER)
                                    .bias_filler(XAVIER)
                    ),
            LayerParameter()
                    .name("sigmoid_layer_02")
                    .type(SIGMOID_LAYER)
                    .input("ip2")
                    .output("s2")
                    .sigmoid_param(
                            SigmoidParameter()
                    ),
            LayerParameter()
                    .name("loss")
                    .type(EUCLIDEAN_LOSS_LAYER)
                    .phase(TRAIN)
                    .input("s2")
                    .input("label")
                    .output("loss")
                    .euclidean_param(
                            EuclideanLossParameter()
                    ),
            LayerParameter()
                    .name("accuracy")
                    .type(ACCURACY_LAYER)
                    .phase(TEST)
                    .input("s2")
                    .input("label")
                    .output("accuracy")
                    .accuracy_param(
                            AccuracyParameter()
                    )
    };

    auto optimize_param = OptimizerParameter()
            .mode(Global::CPU)
            .max_iter(20000)
            .test_iter(100)
            .test_interval(500)
            .train_net_param(NetworkParameter().layer_params(params))
            .test_net_param(NetworkParameter().layer_params(params));

    SgdOptimizer<CPU, float> o(optimize_param);

    o.optimize();

    return 0;
}
