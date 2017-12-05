#include <cstring>
#include "zmatrix.h"


int main()
{
    // 2 x 2, 3 channels
    z::Matrix m1(2, 3, 3);
    z::Matrix m2(2, 3, 3, 2.9);
    z::Matrix m3(2, 3, 3, z::Scalar{ 10, 50, 30 });
    z::Matrix m4({2, 3, 3}, 2.9);
    z::Matrix m5({2, 3, 3}, z::Scalar{ 10, 50, 30 });
    std::cout << m1 << m2 << m3 << m4 << m5;

    // matlab
    auto mm1 = z::Matrix::zeros(2, 3, 3);
    auto mm2 = z::Matrix::zeros({2, 3, 3});
    auto mm3 = z::Matrix::eye(3, 3);
    auto mm4 = z::Matrix::eye({3, 3, 1});
    auto mm5 = z::Matrix::ones(2, 3, 3);
    auto mm6 = z::Matrix::ones({2, 3, 3});
    std::cout << mm1 << mm2 << mm3 << mm4 << mm5 << mm6;

    //
    z::Matrix ml1 = { 1, 2, 3, 4 };
    z::Matrix ml2({1, 2, 3});
    std::cout << ml1 << ml2;

    // random
    std::default_random_engine random_engine(time(nullptr));
    std::uniform_real_distribution<double> real_distribution(-1.0, 1.0);

    z::Matrix64f mr1({4, 3, 3}, std::make_pair(random_engine, real_distribution));
    std::cout << mr1;

    // Matrix
    z::_Matrix<z::_Matrix<double>> mmd1({2, 2, 1}, z::_Matrix<double>({2, 2, 2}, z::Scalar{2.6, 1.4}));
    z::_Matrix<z::_Matrix<z::_Matrix<double>>> mmd2({3, 3, 1}, mmd1);

    return 0;
}