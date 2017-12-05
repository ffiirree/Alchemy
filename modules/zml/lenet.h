#ifndef _ZML_LENET_H
#define _ZML_LENET_H

#include <cstdint>
#include <zcore/matrix.h>

namespace z {

class LeNet{
    using Pair = std::pair<Matrix, uint8_t>;
public:
    LeNet() = default;

    void run(std::vector<Pair>& training_data, const std::vector<Pair>& test_data);

    void training(const std::vector<Pair>& batch);
    std::vector<std::vector<LeNet::Pair>> split(const std::vector<LeNet::Pair> &training_data, int size) const;

    _Matrix<double> convolution(const _Matrix<double>& InputMatrix, const _Matrix<double>& kernel);
    _Matrix<double> max_pooling(const _Matrix<double>& InputMatrix);
    _Matrix<double> average_pooling(const _Matrix<double>& InputMatrix);
private:
};

} //! namespace z

#endif //!_ZML_LENET_H