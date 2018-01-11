#include <zmatrix.h>
#include "zml/network.hpp"
#include "zml/optimize.hpp"

using namespace z;
using namespace std;

int main()
{
    MnistLoader train_loader("/home/ffiirree/Code/zMatrix/resources/mnist/train-images.idx3-ubyte",
                             "/home/ffiirree/Code/zMatrix/resources/mnist/train-labels.idx1-ubyte");

    MnistLoader test_loader("/home/ffiirree/Code/zMatrix/resources/mnist/t10k-images.idx3-ubyte",
                            "/home/ffiirree/Code/zMatrix/resources/mnist/t10k-labels.idx1-ubyte");


    vector<LayerParameter> params = {
            LayerParameter()
                    .name("train input layer")
                    .type(INPUT_LAYER)
                    .phase(TRAIN)
                    .output("data")
                    .output("label")
                    .input_param(
                            InputParameter()
                                    .source(train_loader)
                                    .batch_size(10)
                                    .scale(1./255)
                    ),
            LayerParameter()
                    .name("test input layer")
                    .type(INPUT_LAYER)
                    .phase(TEST)
                    .output("data")
                    .output("label")
                    .input_param(
                            InputParameter()
                                    .source(test_loader)
                                    .batch_size(100)
                                    .scale(1./255)
                    ),
            LayerParameter()
                    .name("conv layer 1")
                    .type(CONVOLUTION_LAYER)
                    .input("data")
                    .output("conv1")
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
                    .name("pooling layer 1")
                    .type(POOLING_LAYER)
                    .input("conv1")
                    .output("p1")
                    .pooling_param(
                            PoolingParameter()
                                    .kernel_size(2)
                                    .stride(2)
                                    .type(MAX)
                    ),
            LayerParameter()
                    .name("ip1")
                    .type(INNER_PRODUCT_LAYER)
                    .input("p1")
                    .output("ip1")
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
                    .name("sigmoid layer 1")
                    .type(SIGMOID_LAYER)
                    .input("ip1")
                    .output("s1")
                    .sigmoid_param(
                            SigmoidParameter()
                    ),
            LayerParameter()
                    .name("ip2")
                    .type(INNER_PRODUCT_LAYER)
                    .input("s1")
                    .output("ip2")
                    .ip_param(
                            InnerProductParameter()
                                    .output_size(10)
                                    .wlr(0.15)
                                    .blr(0.3)
                                    .weight_filler(XAVIER)
                                    .bias_filler(XAVIER)
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

    auto optimize_param = OptimizeParameter()
            .max_iter(60000)
            .test_iter(100)
            .test_interval(500)
            .train_net_param(NetworkParameter().layer_params(params))
            .test_net_param(NetworkParameter().layer_params(params));

    Optimize<float> o(optimize_param);

    o.run();

    return 0;
}
