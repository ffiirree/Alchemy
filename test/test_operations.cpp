#include "zmatrix.h"

int main()
{
    auto image = z::imread("test.jpeg");
    auto roi = image(z::Rect(50, 50, 10, 3));

    std::cout << roi << std::endl;

    std::cout << roi + roi << std::endl;
    std::cout << roi - roi << std::endl;

    std::cout << roi + z::Scalar{10, 20, 50} << std::endl;
    std::cout << roi - z::Scalar{10, 20, 50} << std::endl;

    std::cout << roi * 0.5 << std::endl;
    std::cout << roi / 0.5 << std::endl;

    std::cout << (roi == z::Scalar{100, 100, 100}) << std::endl;
    std::cout << (roi > 100) << std::endl;

    std::cout << z::abs(z::Matrix8s(3, 5, 3, z::Scalar{ -1, 2, 255})) << std::endl;
    z::Matrix _r;
    z::absdiff(roi, z::Matrix(3, 10, 3, z::Scalar{ 0, 2, 255}), _r);
    std::cout << _r << std::endl;

    z::Matrix add_r;
    z::Matrix mask(3, 10, 3, 0);
    auto mask_roi = mask(z::Rect(1, 1, 2, 2));
    mask_roi.fill(z::Scalar{ 1, 1, 1 });
    std::cout << mask << std::endl;
    z::add(roi, z::Matrix(3, 10, 3, z::Scalar{100, 100, 100}), add_r, mask);
    std::cout << add_r << std::endl;

    return 0;
}