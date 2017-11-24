#include "zmatrix.h"

int main()
{
    auto image = z::imread("test.jpeg");
    z::Matrix gray;
    z::cvtColor(image, gray, z::BGR2GRAY);

    z::Matrix binary_image, binary_inv_image, trunc_image, tozero_image, tozero_inv_image;
    z::threshold(gray, binary_image, 150, 250, z::THRESH_BINARY);
    z::threshold(gray, binary_inv_image, 150, 250, z::THRESH_BINARY_INV);
    z::threshold(gray, trunc_image, 150, 250, z::THRESH_TRUNC);
    z::threshold(gray, tozero_image, 150, 250, z::THRESH_TOZERO);
    z::threshold(gray, tozero_inv_image, 150, 250, z::THRESH_TOZERO_INV);


    z::imshow("原图", image);
    z::imshow("二值阈值", binary_image);
    z::imshow("反二值", binary_inv_image);
    z::imshow("截断阈值", trunc_image);
    z::imshow("归零", tozero_image);
    z::imshow("反归零", tozero_inv_image);
    z::waitKey(0);

    return 0;
}