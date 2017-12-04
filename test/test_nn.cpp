#include <zml/network.h>
#include "zmatrix.h"
#include <random>

using namespace z;
using namespace std;

int main()
{
    z::MnistLoader loader("/home/ffiirree/Code/zMatrix/resources/mnist/train-images.idx3-ubyte",
            "/home/ffiirree/Code/zMatrix/resources/mnist/train-labels.idx1-ubyte");

	z::MnistLoader loader_test("/home/ffiirree/Code/zMatrix/resources/mnist/t10k-images.idx3-ubyte",
		"/home/ffiirree/Code/zMatrix/resources/mnist/t10k-labels.idx1-ubyte");

    z::Network net({784, 15, 10});
    auto data = loader.data();
    net.stochastic_gradient_descent(data, 2.0, 30, loader_test.data());
    return 0;
}