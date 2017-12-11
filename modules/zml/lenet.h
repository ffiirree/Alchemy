#ifndef _ZML_LENET_H
#define _ZML_LENET_H

#include <cstdint>
#include <zcore/matrix.h>

namespace z {

class LeNet{
    using Pair = std::pair<Matrix, uint8_t>;
public:
    LeNet();

    void run(std::vector<Pair>& training_data, const std::vector<Pair>& test_data);

    void training(const std::vector<Pair>& batch);
    std::vector<std::vector<LeNet::Pair>> split(const std::vector<LeNet::Pair> &training_data, int size) const;

    _Matrix<double> pp(const _Matrix<uint8_t>& input);
    _Matrix<double> convolution(const _Matrix<double>& InputMatrix, const _Matrix<double>& kernel, const double bias);
    _Matrix<double> convolution(const _Matrix<_Matrix<double>>& InputMatrixs, const _Matrix<double>& kernel, const double bias);
    _Matrix<double> max_pooling(const _Matrix<double>& InputMatrix, double c, double b);
    _Matrix<double> average_pooling(const _Matrix<double>& InputMatrix, double c, double b);
private:
    double sigmoid(double z);
    double dsigmoid(double z);

    _Matrix<double> sigmoid(const _Matrix<double>& z);
    _Matrix<double> dsigmoid(const _Matrix<double>& z);

    double tanh(double z);
    double dtanh(double z);

    _Matrix<double> tanh(const _Matrix<double>& z);
    _Matrix<double> dtanh(const _Matrix<double>& z);



    // Input layer
    _Matrix<double> input_data;

    // C1
    _Matrix<_Matrix<double>> c1_data;
    _Matrix<_Matrix<double>> c1_kernels;
    _Matrix<double> c1_biases;

    // S2
    _Matrix<_Matrix<double>> s2_data;
    _Matrix<double> s2_coefficients;
    _Matrix<double> s2_biases;

    // C3
    _Matrix<_Matrix<double>> c3_data;
    _Matrix<_Matrix<double>> c3_kernels;
    _Matrix<double> c3_biases;

    // S4
    _Matrix<_Matrix<double>> s4_data;
    _Matrix<double> s4_coefficients;
    _Matrix<double> s4_biases;

    // C5
    _Matrix<_Matrix<double>> c5_data;
    _Matrix<_Matrix<double>> c5_kernels;
    _Matrix<double> c5_biases;

    // F6
    _Matrix<double> f6_data;
    _Matrix<_Matrix<double>> f6_weights;
    _Matrix<double> f6_biases;

    // G7

};

} //! namespace z

#endif //!_ZML_LENET_H