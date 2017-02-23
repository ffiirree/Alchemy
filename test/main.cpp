#include <iostream>
#include <opencv2\opencv.hpp>
#include <fstream>

#include "zmatrix.h"

int main(int argc, char *argv[])
{
    auto test = z::imread("test.jpeg");
    z::Matrix dst1;
    z::pyrDown(test, dst1);


    z::Matrix dst2;
    z::pyrDown(dst1, dst2);

    z::Matrix dst3;
    z::pyrDown(dst2, dst3);


    z::Matrix updst1;
    z::pyrUp(dst3, updst1);

    z::Matrix updst2;
    z::pyrUp(updst1, updst2);

    cv::imshow("updst2", cv::Mat(updst2));
    cv::waitKey(0);


    return 0;
}