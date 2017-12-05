#include "lenet.h"

namespace z{
void LeNet::run(std::vector<LeNet::Pair> &training_data, const std::vector<LeNet::Pair> &test_data)
{
    // shuffle the training_data
    std::shuffle(training_data.begin(), training_data.end(), std::default_random_engine(time(nullptr)));

    auto mini_batches = split(training_data, 10);
    auto index = 0;
    for(auto& batch : mini_batches) {
        training(batch);
        if (index++ % 100 == 0) std::cout << '.';
    }

}

void z::LeNet::training(const std::vector<z::LeNet::Pair> &batch)
{
    for(auto &item : batch) {

        // convolution 1
        _Matrix<double> kernel({5, 5, 1}, 1.0);
        auto out = convolution(item.first, kernel);
    }
}


std::vector<std::vector<LeNet::Pair>> LeNet::split(const std::vector<LeNet::Pair> &training_data, int size) const
{
    std::vector<std::vector<LeNet::Pair>> out;
    for (size_t i = 0; i < training_data.size(); i += size) {
        out.emplace_back(training_data.begin() + i, training_data.begin() + std::min(training_data.size(), i + size));
    }
    return out;
}

_Matrix<double> LeNet::convolution(const _Matrix<double> &InputMatrix,  const _Matrix<double>& kernel)
{
    assert(kernel.rows % 2 == 1);
    assert(kernel.cols % 2 == 1);

    auto _rows = InputMatrix.rows - kernel.rows / 2;
    auto _cols = InputMatrix.cols - kernel.cols / 2;

    _Matrix<double> OutputMatrix(_rows, _cols);

    for(auto i = kernel.rows / 2; i < _rows; ++i) {
        for(auto j = kernel.cols / 2; j < _cols; ++j) {

            double temp = 0;

            for(auto ki = 0; ki < kernel.rows; ++ki) {
                for(auto kj = 0; kj < kernel.cols; ++kj) {
                    temp += InputMatrix.at(i - kernel.rows/2 + ki, j - kernel.cols/2 + kj) * kernel.at(ki, kj);
                }
            }
            OutputMatrix.at(i, j) = temp;
        }
    }

    return _Matrix<double>();
}

} //! namespace z

