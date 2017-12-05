#include "zmatrix.h"

int main()
{
    // Load data.
    z::MnistLoader loader("D:\\zMatrix\\resources\\mnist\\train-images.idx3-ubyte",
                          "D:\\zMatrix\\resources\\mnist\\train-labels.idx1-ubyte");

    z::MnistLoader loader_test("D:\\zMatrix\\resources\\mnist\\t10k-images.idx3-ubyte",
                               "D:\\zMatrix\\resources\\mnist\\t10k-labels.idx1-ubyte");

    auto training_data = loader.data();
    auto test_data = loader_test.data();

    // LeNet
    z::LeNet lenet;
    lenet.run(training_data, test_data);

}