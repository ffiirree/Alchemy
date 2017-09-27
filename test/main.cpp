#include <iostream>
#include "zmatrix.h"


int main()
{
    auto image = z::imread("test.jpeg");

    cv::Mat a(3, 5, CV_32F, cv::Scalar{ 2 });
    cv::Mat b(5, 2, CV_32F, cv::Scalar{ 3 });

    a *= b;


    z::Matrix32f a1(2, 3, 1);
    z::Matrix32f b1(3, 3, 1);
    a1 = {
        1, 2, 3,
        2, 1, 0
    };

    b1 = {
        1, -2, 0,
        3, -1, 1,
        0, 4, 1
    };

    a1 *= b1;
    
    return 0;
}