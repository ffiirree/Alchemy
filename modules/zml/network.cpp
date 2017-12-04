#include "network.h"
#include <random>
#include <boost/range/combine.hpp>
using std::vector;

namespace z {
Network::Network(const vector<int> &layers)
        :layers_(layers), bs_(layers.size() - 1, 1), ws_(layers.size() - 1, 1), zs_(layers.size() - 1, 1), os_(layers.size(), 1)
{
    assert(!layers.empty());

    std::default_random_engine random_engine(time(nullptr));
    std::uniform_real_distribution<double> real_distribution(-1.0, 1.0);

    // Each layer except input layer.
    for(size_t i = 1; i < layers.size(); ++i) {
        bs_.at(i - 1) = _Matrix<double>(layers[i], 1, 1, std::make_pair(random_engine, real_distribution));
        ws_.at(i - 1) = _Matrix<_Matrix<double>>(layers[i], 1, 1, _Matrix<double>(layers[i - 1], 1, 1, std::make_pair(random_engine, real_distribution)));
    }
}

void Network::feedforward(const Pair& pair)
{
    // Input layer.
    auto t1 = pair.first.reshape(1, pair.first.size());
    auto t = _Matrix<double>(t1) / 255.0;
    os_.at(0) = t;

    // Other layers.
    for(size_t i = 1; i < layers_.size(); ++i) {
        _Matrix<double> zs(layers_.at(i), 1);

        // z = \sum wx + b
        for(auto j = 0; j < layers_.at(i); ++j) {
            zs.at(j) = (os_.at(i - 1).t() * ws_.at(i - 1).at(j)).at(0) + bs_.at(i - 1).at(j);
        }
        zs_.at(i - 1) = zs;
        // o = \sigma(z)
        os_.at(i) = sigmoid(zs);
    }
}

void Network::backpropagation(const Pair& pair, _Matrix<_Matrix<double>>& nabla_b, _Matrix<_Matrix<_Matrix<double>>>& nabla_w)
{
    feedforward(pair);

    // Output layer
    _Matrix<double> goal(layers_.back(), 1, 1, 0);
    goal.at(pair.second) = 1.0;
    auto delta = (os_.at(os_.total() - 1) - goal).mul(sigmoid_prime(zs_.at(zs_.size() - 1)));
    nabla_b.at(1) = delta;

    _Matrix<_Matrix<double>> nw(layers_.back(), 1);
    for(auto i = 0; i < layers_.back(); ++i) {
        nw.at(i) = delta.at(i) * os_.at(os_.size() - 2);
    }
    nabla_w.at(1) = nw;

    // Hidden layer
    for(auto layer = layers_.size() - 2; layer > 0; --layer) {
        delta = (delta.t() * ws_.at(layer)).at(0).mul(sigmoid_prime(zs_.at(zs_.size() - 2)));
        nabla_b.at(layer - 1) = delta;

        _Matrix<_Matrix<double>> nw2(layers_.at(layer), 1);
        for (auto j = 0; j < layers_.at(1); ++j) {
            nw2.at(j) = delta.at(j) * os_.at(os_.size() - 3);
        }
        nabla_w.at(layer - 1) = nw2;
    }
}

void Network::stochastic_gradient_descent(vector<Pair>& training_data, double alpha, int times, const vector<Pair>& test_data)
{
    for(auto i = 0; i < times; ++i) {
        // shuffle the training_data
        std::shuffle(training_data.begin(), training_data.end(), std::default_random_engine(time(nullptr)));

        auto mini_batches = split(training_data, 10);
        auto index = 0;
        for(auto& batch : mini_batches) {
            training(batch, alpha);
            if (index++ % 100 == 0) std::cout << '.';
        }

        // test
        if (test_data.empty()) return;

        uint32_t counter = 0;
        for (const auto& pair : test_data) {
            feedforward(pair);

            if (*std::max_element(std::begin(os_.at(os_.size() - 1)), std::end(os_.at(os_.size() - 1))) == os_.at(os_.size() - 1).at(pair.second))
                counter++;
        }

        std::cout << "Iteration<" << i << ">: " << counter << "/" << test_data.size() << std::endl;
    }
}

void Network::training(const vector<Pair> &batch, double alpha)
{
    _Matrix<_Matrix<double>> nabla_b(bs_.size(), 1);
    _Matrix<_Matrix<_Matrix<double>>> nabla_w(ws_.size(), 1);

    // Init
    for (size_t i = 0; i < bs_.total(); ++i) {
        nabla_b.at(i) = _Matrix<double>(bs_.at(i).shape(), 0);
    }

    for (size_t i = 0; i < ws_.total(); ++i) {
        _Matrix<_Matrix<double>> nw(ws_.at(i).size(), 1);
        for(size_t j = 0; j < ws_.at(i).total(); ++j) {
            nw.at(j) = _Matrix<double>(ws_.at(i).at(j).shape(),  0);
        }
        nabla_w.at(i) = nw;
    }


    for(auto& item: batch) {
        _Matrix<_Matrix<double>> delta_nabla_b(bs_.size(), 1);
        _Matrix<_Matrix<_Matrix<double>>> delta_nabla_w(ws_.size(), 1);
        backpropagation(item, delta_nabla_b, delta_nabla_w);

        for(const auto& b : boost::combine(nabla_b, delta_nabla_b)) {
            b.get<0>() += b.get<1>();
        }

        for(const auto& ws : boost::combine(nabla_w, delta_nabla_w)) {
            for(const auto & w: boost::combine(ws.get<0>(), ws.get<1>())) {
                w.get<0>() += w.get<1>();
            }
        }
    }

    // update
    for (const auto& b : boost::combine(bs_, nabla_b)) {
        b.get<0>() = b.get<0>() - (alpha / batch.size()) * b.get<1>();
    }

    for (const auto& ws : boost::combine(ws_, nabla_w)) {
        for (const auto& w : boost::combine(ws.get<0>(), ws.get<1>())) {
            w.get<0>() = w.get<0>() - (alpha / batch.size()) * w.get<1>();
        }
    }
}



double sigmoid(double z)
{
    return 1.0 / (1.0 + std::exp(-z));
}

double sigmoid_prime(double z)
{
    return sigmoid(z) * (1 - sigmoid(z));
}

_Matrix<double> sigmoid(_Matrix<double> z)
{
    _Matrix<double> _r(z.shape());

    for(const auto& item: boost::combine(_r, z)) {
        item.get<0>() = sigmoid(item.get<1>());
    }
    return _r;
}

_Matrix<double> sigmoid_prime(_Matrix<double> z)
{
    _Matrix<double> _r(z.shape());

    for (const auto& item : boost::combine(_r, z)) {
        item.get<0>() = sigmoid_prime(item.get<1>());
    }
    return _r;
}

vector<vector<Network::Pair>> Network::split(const vector<Pair> &training_data, int size) const
{
    vector<vector<Pair>> out;
    for (size_t i = 0; i < training_data.size(); i += size) {
        out.emplace_back(training_data.begin() + i, training_data.begin() + std::min(training_data.size(), i + size));
    }
    return out;
}

} //! namespace z