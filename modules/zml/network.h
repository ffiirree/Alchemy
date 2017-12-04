#ifndef _ZML_NEURON_H
#define _ZML_NEURON_H

#include <vector>
#include <zcore/matrix.h>

namespace z {

class Network {
    using Pair = std::pair<Matrix, uint8_t>;
public:
    Network(const std::vector<int>& layers);
    Network(const Network&) = delete;
    Network&operator=(const Network&) = delete;
    ~Network() = default;

    // Stochastic gradient descent
    void stochastic_gradient_descent(std::vector<Pair>& training_data, double alpha, int times, const std::vector<Pair>& test_data = std::vector<Pair>());

private:
    void feedforward(const Pair& pair);
    void backpropagation(const Pair& pair, _Matrix<_Matrix<double>>& nabla_b, _Matrix<_Matrix<_Matrix<double>>>& nabla_w);

	std::vector<std::vector<Pair>> split(const std::vector<Pair>& training_data, int size) const;
    void training(const std::vector<Pair>& batch, double alpha);

	double alpha_ = 0.0;
	std::vector<int> layers_;

    // Biases
	_Matrix<_Matrix<double>> bs_;
    // Weights
	_Matrix<_Matrix<_Matrix<double>>> ws_;
    // z = \sum wx + b
	_Matrix<_Matrix<double>> zs_;
    // Outputs: o = \sigma(z)
    _Matrix<_Matrix<double>> os_;
};

double sigmoid(double z);
double sigmoid_prime(double z);

_Matrix<double> sigmoid(_Matrix<double> z);
_Matrix<double> sigmoid_prime(_Matrix<double> z);

} //! namespace z

#endif //_ZML_NEURON_H
