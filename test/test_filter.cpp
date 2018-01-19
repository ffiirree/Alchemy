#include <alchemy.h>

int main()
{
    auto image = alchemy::imread("test.jpeg");

    // 线性滤波
    alchemy::Matrix box_image, blur_image, g_image;
    alchemy::boxFilter(image, box_image, {5, 5}, true);
    alchemy::blur(image, blur_image, {5, 5});
    alchemy::GaussianBlur(image, g_image, {5, 5});

    // 非线性滤波
    alchemy::Matrix median_image, b_image;
    alchemy::medianFilter(image, median_image, {5 , 5});
    alchemy::bilateralFilter(image, b_image, 25, 25 * 2, 25/2);


    alchemy::imshow("original", image);
    alchemy::imshow("boxFilter", box_image);
    alchemy::imshow("blur", blur_image);
    alchemy::imshow("gaussianBlur", g_image);

    alchemy::imshow("medianFilter", median_image);
    alchemy::imshow("bilateralFilter", b_image);

    alchemy::waitKey(0);

    return 0;
}