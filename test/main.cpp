#include <iostream>
#include <opencv2\opencv.hpp>
#include <fstream>

#include "zmatrix.h"

int main(int argc, char *argv[])
{
    auto test = z::imread("test.jpeg");

    z::imshow("test", test);

    z::Matrix gray;
    z::cvtColor(test, gray, BGR2GRAY);

    z::imshow("gray", gray);

    z::Matrix bin, bin_inv, trunc, tozero, tozero_inv;

    z::threshold(gray, bin, 150, 200, THRESH_BINARY);
    z::threshold(gray, bin_inv, 150, 200, THRESH_BINARY_INV);
    z::threshold(gray, trunc, 150, 200, THRESH_TRUNC);
    z::threshold(gray, tozero, 150, 200, THRESH_TOZERO);
    z::threshold(gray, tozero_inv, 150, 200, THRESH_TOZERO_INV);

    z::imshow("bin", bin);
    z::imshow("bin_inv", bin_inv);
    z::imshow("trunc", trunc);
    z::imshow("tozero", tozero);
    z::imshow("tozero_inv", tozero_inv);

    z::waitKey(0);
    /*system("pause");*/
    return 0;
}