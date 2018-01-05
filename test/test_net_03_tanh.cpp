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
                                    .batch_size(10)//暂时需要和训练一致，在ip层有参数与之相关
                                    .scale(1./255)
                    ),
            LayerParameter()
                    .name("ip layer 1")
                    .type(INNER_PRODUCT_LAYER)
                    .input("data")
                    .output("ip1")
                    .ip_param(
                            InnerProductParameter()
                                    .neuron_size(30)
                                    .wlr(0.1)
                                    .blr(0.1)
                                    .weight_filler(XAVIER)
                                    .bias_filler(XAVIER)
                    ),
            LayerParameter()
                    .name("tanh layer 1")
                    .type(TANH_LAYER)
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
                                    .neuron_size(10)
                                    .wlr(0.1)
                                    .blr(0.1)
                                    .weight_filler(XAVIER)
                                    .bias_filler(XAVIER)
                    ),
            LayerParameter()
                    .name("tanh layer 2")
                    .type(TANH_LAYER)
                    .input("ip2")
                    .output("s2")
                    .sigmoid_param(
                            SigmoidParameter()
                    ),
            LayerParameter()
                    .name("eucl layer")
                    .type(EUCLIDEAN_LOSS_LAYER)
                    .phase(TRAIN)
                    .input("s2")
                    .input("label")
                    .output("loss")
                    .euclidean_param(
                            EuclideanLossParameter()
                    ),
            LayerParameter()
                    .name("acc layer")
                    .type(ACCURACY_LAYER)
                    .phase(TEST)
                    .input("s2")
                    .input("label")
                    .output("accuracy")
                    .accuracy_param(
                            AccuracyParameter()
                    )
    };

    auto optimize_param = OptimizeParameter()
            .epochs(10)
            .train_net_param(NetworkParameter().layer_params(params))
            .test_net_param(NetworkParameter().layer_params(params));

    Optimize<double> o(optimize_param);

    o.run();

    return 0;
}
