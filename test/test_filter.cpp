#include "zmatrix.h"

int main()
{
    auto image = z::imread("test.jpeg");

    // 线性滤波
    z::Matrix box_image, blur_image, g_image;
    z::boxFilter(image, box_image, {5, 5}, true);
    z::blur(image, blur_image, {5, 5});
    z::GaussianBlur(image, g_image, {5, 5});

    // 非线性滤波
    z::Matrix median_image, b_image;
    z::medianFilter(image, median_image, {5 , 5});
    z::bilateralFilter(image, b_image, 25, 25 * 2, 25/2);


    z::imshow("原图", image);
    z::imshow("方框滤波", box_image);
    z::imshow("均值滤波", blur_image);
    z::imshow("高斯滤波", g_image);

    z::imshow("中值滤波", median_image);
    z::imshow("双边滤波", b_image);

    z::waitKey(0);

    return 0;
}