#include <iostream>
#include <opencv2\opencv.hpp>
#include <fstream>

#include "zmatrix.h"

int main(int argc, char *argv[])
{
    auto test = z::imread("test.jpeg");
    z::Matrix dst;

    TimeStamp timer;
    timer.start();
    z::bilateralFilter(test, dst, 25, 50, 15);
    std::cout << "z: " << timer.runtime() << std::endl;

    cv::Mat cvt = test;
    cv::Mat cvdst;

    timer.start();
    cv::bilateralFilter(cvt, cvdst, 25, 50, 15);
    std::cout << "cv: " << timer.runtime() << std::endl;

    system("pause");
    return 0;
}