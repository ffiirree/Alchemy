#include <iostream>
#include <opencv2\opencv.hpp>
#include <fstream>

#include "zmatrix.h"

int main(int argc, char *argv[])
{
    auto test = z::imread("test.jpeg");
    z::Matrix zgray;
    z::cvtColor(test, zgray, BGR2GRAY);
    z::Matrix zldst;
    z::Laplacian(zgray, zldst, 3);


    //
    cv::Mat cvt = test;

    cv::Mat gray;
    cv::cvtColor(cvt, gray, CV_BGR2GRAY);

    cv::Mat ldst;
    cv::Laplacian(gray, ldst, 8, 3);
    return 0;
}