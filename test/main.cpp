#include <iostream>
#include <opencv2\opencv.hpp>
#include <fstream>

#include "zmatrix.h"

int main(int argc, char *argv[])
{
    auto test = z::imread("test.jpeg");
    z::Matrix zhsv, zhsi;
    z::cvtColor(test, zhsv, BGR2HSV);
    z::cvtColor(test, zhsi, BGR2HSI);
    z::imshow("zhsv", zhsv);
    z::imshow("zhsi", zhsi);
    z::waitKey(0);
    return 0;
}