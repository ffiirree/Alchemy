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
    z::differenceOfGaussian(test, dst, { 5, 5 }, 0.6, 2.5);
    std::cout << timer.runtime() << std::endl;


    // 
    cv::Mat cvt = cv::imread("test.jpeg");
    cv::Mat cvdst1, cvdst2, cvres;
    cv::GaussianBlur(cvt, cvdst1, cv::Size(5, 5), 0.6, 0.6);
    cv::GaussianBlur(cvt, cvdst2, cv::Size(5, 5), 2.5, 2.5);

    cvres = cvdst1 - cvdst2;

    return 0;
}