#include "zmatrix.h"

int main()
{
    auto image = z::imread("test.jpeg");

    // 膨胀和腐蚀
    z::Matrix erode_image, dilate_image;
    z::erode(image, erode_image, {5, 5});
    z::dilate(image, dilate_image, {5, 5});

    // 其他
    z::Matrix opening_image, closing_image, th_image, b_image;
    z::morphEx(image, opening_image, z::MORP_OPEN, {5, 5});
    z::morphEx(image, closing_image, z::MORP_CLOSE, {5, 5});
    z::morphEx(image, th_image, z::MORP_TOPHAT, {5, 5});
    z::morphEx(image, b_image, z::MORP_BLACKHAT, {5, 5});

    z::imshow("原图", image);
    z::imshow("腐蚀", erode_image);
    z::imshow("膨胀", dilate_image);
    z::imshow("开运算", opening_image);
    z::imshow("闭运算", closing_image);
    z::imshow("顶帽", th_image);
    z::imshow("黑帽", b_image);
    z::waitKey(0);

    return 0;
}