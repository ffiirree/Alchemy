#include <alchemy.h>

int main()
{
    auto image = alchemy::imread("../resources/test.jpeg");
    alchemy::Matrix gray;
    alchemy::cvtColor(image, gray, alchemy::BGR2GRAY);

    alchemy::Matrix binary_image, binary_inv_image, trunc_image, tozero_image, tozero_inv_image;
    alchemy::threshold(gray, binary_image, 150, 250, alchemy::THRESH_BINARY);
    alchemy::threshold(gray, binary_inv_image, 150, 250, alchemy::THRESH_BINARY_INV);
    alchemy::threshold(gray, trunc_image, 150, 250, alchemy::THRESH_TRUNC);
    alchemy::threshold(gray, tozero_image, 150, 250, alchemy::THRESH_TOZERO);
    alchemy::threshold(gray, tozero_inv_image, 150, 250, alchemy::THRESH_TOZERO_INV);


    alchemy::imshow("原图", image);
    alchemy::imshow("二值阈值", binary_image);
    alchemy::imshow("反二值", binary_inv_image);
    alchemy::imshow("截断阈值", trunc_image);
    alchemy::imshow("归零", tozero_image);
    alchemy::imshow("反归零", tozero_inv_image);
    alchemy::waitKey(0);

    return 0;
}