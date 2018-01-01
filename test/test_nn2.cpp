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

    vector<shared_ptr<Layer<double>>> layers;
    layers.push_back(shared_ptr<Layer<double>>(new InputLayer<double>(10, 1, 784, 1)));
    layers.push_back(shared_ptr<Layer<double>>(new InnerProductLayer<double>(10, 1, 15, 1)));
    layers.push_back(shared_ptr<Layer<double>>(new SigmoidLayer<double>(10, 1, 15, 1)));
    layers.push_back(shared_ptr<Layer<double>>(new InnerProductLayer<double>(10, 1, 10, 1)));
    layers.push_back(shared_ptr<Layer<double>>(new SigmoidLayer<double>(10, 1, 10, 1)));
    layers.push_back(shared_ptr<Layer<double>>(new EuclideanLossLayer<double>(10, 1, 1, 1)));

    auto td = train_loader.data();
    Network<double> net(layers, td, test_loader.data());
    net.sgd(5);

    return 0;
}