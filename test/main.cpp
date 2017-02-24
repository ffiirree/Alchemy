#include <iostream>
#include <opencv2\opencv.hpp>
#include <fstream>

#include "zmatrix.h"

int main(int argc, char *argv[])
{
    auto test = z::imread("test.jpeg");

    z::Matrix gray;
    z::cvtColor(test, gray, BGR2GRAY);

    z::Matrix boxdst;
    z::boxFilter(test, boxdst, { 5, 5 }, true);

    z::Matrix dst;


    z::differenceOfGaussian(gray, dst, {5, 5}, 0.6, 0.7);
    
   
    system("pause");
    return 0;
}