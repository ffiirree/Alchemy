#include "lenet.h"
#include <random>
#include <boost/range/combine.hpp>
#include "zimgproc/zimgproc.hpp"

namespace z{
LeNet::LeNet()
{
    std::default_random_engine engine(time(nullptr));
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    // C1: 6@5x5
    c1_data = _Matrix<_Matrix<double>>(6, 1, 1);
    c1_kernels = _Matrix<_Matrix<double>>({ 6, 1, 1 }, _Matrix<double>{ {5, 5, 1}, std::make_pair(engine, distribution)});
    c1_biases = _Matrix<double>({ 6, 1, 1 }, std::make_pair(engine, distribution));

    // S2 6 
    s2_data = _Matrix<_Matrix<double>>(6, 1, 1);
    s2_coefficients = _Matrix<double>({ 6, 1, 1 }, std::make_pair(engine, distribution));
    s2_biases = _Matrix<double>({ 6, 1, 1 }, std::make_pair(engine, distribution));

    // C3 16@5x5
    c3_data = _Matrix<_Matrix<double>>(16, 1, 1);
    c3_kernels = _Matrix<_Matrix<double>>({ 16, 1, 1 }, _Matrix<double>{ {5, 5, 1}, std::make_pair(engine, distribution)});
    c3_biases = _Matrix<double>({ 16, 1, 1 }, std::make_pair(engine, distribution));

    // S4 16
    s4_data = _Matrix<_Matrix<double>>(16, 1, 1);
    s4_coefficients = _Matrix<double>({ 16, 1, 1 }, std::make_pair(engine, distribution));
    s4_biases = _Matrix<double>({ 16, 1, 1 }, std::make_pair(engine, distribution));

    // C5 120@5x5
    c5_data = _Matrix<_Matrix<double>>(120, 1, 1);
    c5_kernels = _Matrix<_Matrix<double>>({ 120, 1, 1 }, _Matrix<double>{ {5, 5, 1}, std::make_pair(engine, distribution)});
    c5_biases = _Matrix<double>({ 120, 1, 1 }, std::make_pair(engine, distribution));

    // F6 84
    f6_data = _Matrix<double>(120, 1, 1);
    f6_weights = _Matrix<_Matrix<double>>({84, 1, 1}, _Matrix<double>({ 120, 1, 1 }, std::make_pair(engine, distribution)));
    f6_biases = _Matrix<double>({ 84, 1, 1 }, std::make_pair(engine, distribution));

    //
}

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

        //// Input layer
        //const auto _in = pp(item.first);

        //// C1: convolution 1
        //for(auto& conv : boost::combine(c1_kernels, c1_biases, c1_data)) {
        //    conv.get<2>() = convolution(_in, conv.get<0>(), conv.get<1>());
        //}

        //// P2: pooling 1
        //for(auto& p: boost::combine(c1_data, s2_coefficients, s2_biases, s2_data)) {
        //    p.get<3>() = max_pooling(p.get<0>(), p.get<1>(), p.get<2>());
        //}

        //// C3: convolution 2
        //// 0-5
        //for(auto i = 0; i < 6; ++i) {
        //    const _Matrix<_Matrix<double>> maps = { s2_data(i), s2_data((i + 1)%6), s2_data((i + 2)%6) };
        //    c3_data(i) = convolution(maps, c3_kernels(i), c3_biases(i));
        //}
        //// 6-11
        //for (auto i = 6; i < 12; ++i) {
        //    const _Matrix<_Matrix<double>> maps = { s2_data(i%6), s2_data((i + 1) % 6), s2_data((i + 2) % 6), s2_data((i + 3) % 6) };
        //    c3_data(i) = convolution(maps, c3_kernels(i), c3_biases(i));
        //}
        //// 12-14
        //for (auto i = 12; i < 15; ++i) {
        //    const _Matrix<_Matrix<double>> maps = { s2_data(i % 6), s2_data((i + 1) % 6), s2_data((i + 3) % 6), s2_data((i + 4) % 6) };
        //    c3_data(i) = convolution(maps, c3_kernels(i), c3_biases(i));
        //}
        //// 15
        //c3_data(15) = convolution(s2_data, c3_kernels(15), c3_biases(15));

        //// P4: pooling 2
        //for (auto& p : boost::combine(c3_data, s4_coefficients, s4_biases, s4_data)) {
        //    p.get<3>() = max_pooling(p.get<0>(), p.get<1>(), p.get<2>());
        //}

        //// C5: convolution 3
        //for (auto& conv : boost::combine(c5_kernels, c5_biases, c5_data)) {
        //    conv.get<2>() = convolution(s4_data, conv.get<0>(), conv.get<1>());
        //}

        //// F6
        //for(auto& n: boost::combine(f6_weights, f6_biases, f6_data)) {
        //    double temp = 0;
        //    for (auto& t : boost::combine(n.get<0>(), c5_data)) {
        //        temp += t.get<0>() * t.get<1>().at(0);
        //    }
        //    n.get<2>() = tanh(temp + n.get<1>());
        //}

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

_Matrix<double> LeNet::pp(const _Matrix<uint8_t>& input)
{
    _Matrix<double> r;
    copyMakeBorder(_Matrix<double>(input) / 225.0, r, 2, 2, 2, 2);
    return r;
}

_Matrix<double> LeNet::convolution(const _Matrix<double> &InputMatrix,  const _Matrix<double>& kernel, const double bias)
{
    assert(kernel.rows % 2 == 1);
    assert(kernel.cols % 2 == 1);

    const auto _rows = InputMatrix.rows - kernel.rows + 1;
    const auto _cols = InputMatrix.cols - kernel.cols + 1;

    _Matrix<double> OutputMatrix(_rows, _cols);

    for(auto i = kernel.rows / 2; i < _rows; ++i) {
        for(auto j = kernel.cols / 2; j < _cols; ++j) {

            double temp = 0;

            for(auto ki = 0; ki < kernel.rows; ++ki) {
                for(auto kj = 0; kj < kernel.cols; ++kj) {
                    temp += InputMatrix.at(i - kernel.rows/2 + ki, j - kernel.cols/2 + kj) * kernel.at(ki, kj);
                }
            }
            OutputMatrix.at(i, j) = temp + bias;
        }
    }

    return OutputMatrix;
}

_Matrix<double> LeNet::convolution(const _Matrix<_Matrix<double>>& InputMatrixs, const _Matrix<double>& kernel, const double bias)
{
    assert(kernel.rows % 2 == 1);
    assert(kernel.cols % 2 == 1);

    const auto _rows = InputMatrixs.at(0).rows - kernel.rows + 1;
    const auto _cols = InputMatrixs.at(0).cols - kernel.cols + 1;

    _Matrix<double> OutputMatrix(_rows, _cols);

    for (auto i = kernel.rows / 2; i < _rows; ++i) {
        for (auto j = kernel.cols / 2; j < _cols; ++j) {

            double temp = 0;
            // 5 x 5
            for (auto ki = 0; ki < kernel.rows; ++ki) {
                for (auto kj = 0; kj < kernel.cols; ++kj) {
                    auto _pos_i = i - kernel.rows / 2 + ki;
                    auto _pos_j = j - kernel.cols / 2 + kj;

                    // Each matrix
                    for(auto& m: InputMatrixs) {
                        temp += m.at(i, j) * kernel.at(ki, kj);
                    }
                }
            }
            OutputMatrix.at(i, j) = temp + bias;
        }
    }

    return OutputMatrix;
}

_Matrix<double> LeNet::max_pooling(const _Matrix<double>& InputMatrix, double c, double b)
{
    _Matrix<double> OutputMatrix(InputMatrix.rows/2, InputMatrix.cols/2);

    for(auto i = 0; i < OutputMatrix.rows; ++i) {
        for(auto j = 0; j < OutputMatrix.cols; ++j) {

            const auto _row = 2 * i;
            const auto _col = 2 * j;
            const auto _max = std::max(
                std::max(InputMatrix(_row, _col), InputMatrix(_row + 1, _col)),
                std::max(InputMatrix(_row, _col + 1), InputMatrix(_row + 1, _col + 1))
            );
            OutputMatrix(i, j) = tanh(_max * c + b);
        }
    }

    return OutputMatrix;
}

_Matrix<double> LeNet::average_pooling(const _Matrix<double>& InputMatrix, double c, double b)
{
    _Matrix<double> OutputMatrix(InputMatrix.rows / 2, InputMatrix.cols / 2);

    for (auto i = 0; i < OutputMatrix.rows; ++i) {
        for (auto j = 0; j < OutputMatrix.cols; ++j) {

            const auto _row = 2 * i;
            const auto _col = 2 * j;
            const auto _ave = (InputMatrix(_row, _col) + InputMatrix(_row + 1, _col) + InputMatrix(_row, _col + 1) + InputMatrix(_row + 1, _col + 1)) / 4.0;
            OutputMatrix(i, j) = tanh(_ave * c + b);
        }
    }

    return OutputMatrix;
}

double LeNet::sigmoid(double z)
{
    return 1.0 / (1.0 + std::exp(-z));
}

double LeNet::dsigmoid(double z)
{
    return sigmoid(z) * (1.0 - sigmoid(z));
}

_Matrix<double> LeNet::sigmoid(const _Matrix<double>& z)
{
    _Matrix<double> _r(z.shape());
    for(auto & item: boost::combine(z, _r)) {
        item.get<1>() = sigmoid(item.get<0>());
    }
    return _r;
}

_Matrix<double> LeNet::dsigmoid(const _Matrix<double>& z)
{
    _Matrix<double> _r(z.shape());
    for (auto & item : boost::combine(z, _r)) {
        item.get<1>() = dsigmoid(item.get<0>());
    }
    return _r;
}

double LeNet::tanh(double z)
{
    const auto _1 = exp(z);
    const auto _2 = exp(-z);

    return (_1 - _2) / (_1 + _2);
}

double LeNet::dtanh(double z)
{
    return 1 - tanh(z) * tanh(z);
}

_Matrix<double> LeNet::tanh(const _Matrix<double>& z)
{
    _Matrix<double> _r(z.shape());
    for (auto & item : boost::combine(z, _r)) {
        item.get<1>() = tanh(item.get<0>());
    }
    return _r;
}

_Matrix<double> LeNet::dtanh(const _Matrix<double>& z)
{
    _Matrix<double> _r(z.shape());
    for (auto & item : boost::combine(z, _r)) {
        item.get<1>() = dtanh(item.get<0>());
    }
    return _r;
}
} //! namespace z

