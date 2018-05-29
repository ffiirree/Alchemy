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
                    .name("train input layer")
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
                    .name("simples input layer")
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
                    .name("ip layer 1")
                    .type(INNER_PRODUCT_LAYER)
                    .input("data")
                    .output("ip1")
                    .ip_param(
                            InnerProductParameter()
                                    .output_size(100)
                                    .wlr(0.1)
                                    .blr(0.2)
                                    .weight_filler(XAVIER)
                                    .bias_filler(CONSTANT)
                    ),
            LayerParameter()
                    .name("sigmoid layer 1")
                    .type(SIGMOID_LAYER)
                    .input("ip1")
                    .output("s1")
                    .sigmoid_param(
                            SigmoidParameter()
                    ),
            LayerParameter()
                    .name("ip layer 2")
                    .type(INNER_PRODUCT_LAYER)
                    .input("s1")
                    .output("ip2")
                    .ip_param(
                            InnerProductParameter()
                                    .output_size(10)
                                    .wlr(0.1)
                                    .blr(0.2)
                                    .weight_filler(XAVIER)
                                    .bias_filler(CONSTANT)
                    ),
            LayerParameter()
                    .name("scel layer")
                    .type(SIGMOID_CROSS_ENTORPY_LOSS_LAYER)
                    .phase(TRAIN)
                    .input("ip2")
                    .input("label")
                    .output("loss")
                    .euclidean_param(
                            EuclideanLossParameter()
                    ),
            LayerParameter()
                    .name("acc layer")
                    .type(ACCURACY_LAYER)
                    .phase(TEST)
                    .input("ip2")
                    .input("label")
                    .output("accuracy")
                    .accuracy_param(
                            AccuracyParameter()
                    )
    };

    auto optimize_param = OptimizerParameter()
            .mode(Global::GPU)
            .max_iter(50000)
            .test_iter(100)
            .test_interval(500)
            .regularization_type(L1)
            .train_net_param(NetworkParameter().layer_params(params))
            .test_net_param(NetworkParameter().layer_params(params));

    SgdOptimizer<GPU, float> o(optimize_param);

    o.optimize();

    return 0;
}
