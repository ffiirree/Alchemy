#include <cstring>
#include "zmatrix.h"


int main(int argc, char* argv[])
{
    const auto ROWS = 5;
    const auto COLS = 5;
    const auto CHS = 3;

    // 1.
    z::Matrix test8u(ROWS, COLS, CHS, z::Scalar(-20, 255, 300));
    typedef z::_Vec<uint8_t, CHS> VT;
    for(auto it = test8u.begin<VT>(), end = test8u.end<VT>(); it != end; ++it) {
        if(*it != VT(0, 255, 255)) {
            return 10;
        }
    }
    auto test8u1 = z::Matrix::zeros(5, 5, 3);
    auto test8u2 = z::Matrix::eye(5, 5, 1);
    auto test8u3 = z::Matrix::ones(5, 5, 3);

    auto test8u4 = z::_Matrix<z::Vec3u8>::ones(5, 5, 1);
    for(auto pix: test8u4) {
        std::cout << pix << ", ";
    }


    // 2.
    z::Matrix16u test16u(5, 5, 3, z::Scalar(-20, 255, 300));
    typedef z::_Vec<uint16_t, 3> VT1;
    for(auto it = test16u.begin<VT1>(), end = test16u.end<VT1>(); it != end; ++it) {
        if(*it != VT1(0, 255, 300)) {
            std::cout << *it ;
            return 20;
        }
    }

    // 3.
    z::Matrix32f test32f(5, 5, 3, z::Scalar(-20, 255, 300));
    typedef z::_Vec<float, 3> VT2;
    for (auto it = test32f.begin<VT2>(), end = test32f.end<VT2>(); it != end; ++it) {
        if (*it != VT2(-20, 255, 300)) {
            std::cout << *it;
            return 30;
        }
    }

    return 0;
}