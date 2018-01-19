#include <alchemy.h>
#include <iostream>

int main()
{
    auto image = alchemy::imread("test.jpeg");
    auto roi = image(alchemy::Rect(50, 50, 10, 3));

    std::cout << roi << std::endl;

    std::cout << roi + roi << std::endl;
    std::cout << roi - roi << std::endl;

    std::cout << roi + alchemy::Scalar{10, 20, 50} << std::endl;
    std::cout << roi - alchemy::Scalar{10, 20, 50} << std::endl;

    std::cout << roi * 0.5 << std::endl;
    std::cout << roi / 0.5 << std::endl;

    std::cout << (roi == alchemy::Scalar{100, 100, 100}) << std::endl;
    std::cout << (roi > 100) << std::endl;

    std::cout << alchemy::abs(alchemy::Matrix8s(3, 5, 3, alchemy::Scalar{ -1, 2, 255})) << std::endl;
    alchemy::Matrix _r;
    alchemy::absdiff(roi, alchemy::Matrix(3, 10, 3, alchemy::Scalar{ 0, 2, 255}), _r);
    std::cout << _r << std::endl;

    alchemy::Matrix add_r;
    alchemy::Matrix mask(3, 10, 3, 0);
    auto mask_roi = mask(alchemy::Rect(1, 1, 2, 2));
    mask_roi.fill(alchemy::Scalar{ 1, 1, 1 });
    std::cout << mask << std::endl;
    alchemy::add(roi, alchemy::Matrix(3, 10, 3, alchemy::Scalar{100, 100, 100}), add_r, mask);
    std::cout << add_r << std::endl;

    return 0;
}