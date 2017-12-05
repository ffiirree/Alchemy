#include "lenet.h"
#include <random>
#include <boost/range/combine.hpp>

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
    const auto KERNEL_ROWS = 5;
    const auto KERNEL_COLS = 5;

    const auto IMAGE_ROWS = batch.at(0).first.rows;
    const auto IMAGE_COLS = batch.at(0).first.cols;

    for(auto &item : batch) {

        // Input layer
        _Matrix<_Matrix<double>> kernels({ 6, 1, 1 }, _Matrix<double>({ KERNEL_ROWS, KERNEL_COLS, 1 }, 1.0 / 25));
        const auto _in = Matrix64f(item.first) / 255.0;

        // C1: convolution 1
        _Matrix<_Matrix<double>> out_1(6, 1, 1);

        for(auto& conv : boost::combine(kernels, out_1)) {
            conv.get<1>() = convolution(_in, conv.get<0>());
        }

        // P2: pooling 1
        _Matrix<_Matrix<double>> out_2(6, 1, 1);
        for(auto& p: boost::combine(out_1, out_2)) {
            p.get<1>() = max_pooling(p.get<0>());
        }

        // C3: convolution 2
        // P4: pooling 2
        // C5: convolution 3
        // F6
        // output layer
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

    return OutputMatrix;
}

_Matrix<double> LeNet::max_pooling(const _Matrix<double>& InputMatrix)
{
    _Matrix<double> OutputMatrix(InputMatrix.rows/2, InputMatrix.cols/2);

    for(auto i = 0; i < OutputMatrix.rows; ++i) {
        for(auto j = 0; j < OutputMatrix.cols; ++j) {

            const auto _row = 2 * i;
            const auto _col = 2 * j;
            OutputMatrix(i, j) = std::max(
                std::max(InputMatrix(_row,      _col), InputMatrix(_row + 1,        _col)),
                std::max(InputMatrix(_row,  _col + 1), InputMatrix(_row + 1,    _col + 1))
            );
        }
    }

    return OutputMatrix;
}

_Matrix<double> LeNet::average_pooling(const _Matrix<double>& InputMatrix)
{
    _Matrix<double> OutputMatrix(InputMatrix.rows / 2, InputMatrix.cols / 2);

    for (auto i = 0; i < OutputMatrix.rows; ++i) {
        for (auto j = 0; j < OutputMatrix.cols; ++j) {

            const auto _row = 2 * i;
            const auto _col = 2 * j;
            OutputMatrix(i, j) = (InputMatrix(_row, _col) + InputMatrix(_row + 1, _col) + InputMatrix(_row, _col + 1) + InputMatrix(_row + 1, _col + 1)) / 4.0;
        }
    }

    return OutputMatrix;
}


} //! namespace z

