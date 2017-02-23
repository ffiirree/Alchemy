#include <iostream>
#include <opencv2\opencv.hpp>
#include <fstream>

#include "zmatrix.h"

int main(int argc, char *argv[])
{
    z::Matrix zm8uc1(3, 3, 1);
    zm8uc1 = {
        3, 2, 1,
        6, 5, 4,
    };

    z::Matrix16s zm16sc1(3, 3, 1);
    zm16sc1 = {
        3, -2, 1,
        6, 5, -4,
        -9, 8, 7
    };

    z::Matrix64f zm64fc3(3, 3, 3);
    zm64fc3 = {
        3.5, 2, 0.6,    5, 6.2, 3.1,     2.1, 2.5, 2.6,
        6, 5, 4,    3.6, 4.6, 8.2,      8.9, 5.2, 1.2,
    };

    cv::Mat cvm8uc1 = zm8uc1;
    cv::Mat cvm16sc1 = zm16sc1;
    cv::Mat cvm64fc3 = zm64fc3;

    z::Matrix16s z8u16sc1;
    z8u16sc1 = zm8uc1;
    z::Matrix16s z64f16sc3;
    z64f16sc3 = zm64fc3;

    auto zm8uc1t = zm8uc1.t();
    auto zm16sc1t = zm16sc1.t();
    auto zm64fc3t = zm64fc3.t();

    z::Matrix zm8uc2(5, 3, 2);
    z::Matrix zm(zm8uc2.size(), zm8uc2.chs);

    system("pause");
    return 0;
}