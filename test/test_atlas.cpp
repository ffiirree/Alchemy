#include <iostream>
#include "zmatrix.h"
#include "zml/math_op.hpp"

using namespace z;
using namespace std;

int main()
{
    z::Matrix64f a(4, 2, 1);
    a = {
            1.5, 3.2,
            3.0, 12.0,
            -2.0, 4.0,
            2.3, 4.2
    };

    z::Matrix64f b(3, 2, 1);
    b = {
            2., 1.0,
            3.0, 2.0,
            1.0, 4.0
    };

    std::cout << a * b.t() << std::endl;

    z::Matrix64f c(4, 3, 1);

    int M = 4;
    int N = 3;
    int K = 2;
    matrix_mul(CblasNoTrans, CblasTrans, M, N, K, 1., a.ptr(), b.ptr(), 0., c.ptr());

    std::cout << c;

    return 0;
}
