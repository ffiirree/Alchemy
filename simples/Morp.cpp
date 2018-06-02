#include <alchemy.h>

int main()
{
    auto image = alchemy::imread("../resources/test.jpeg");

    // 膨胀和腐蚀
    alchemy::Matrix erode_image, dilate_image;
    alchemy::erode(image, erode_image, {5, 5});
    alchemy::dilate(image, dilate_image, {5, 5});

    // 其他
    alchemy::Matrix opening_image, closing_image, th_image, b_image;
    alchemy::morphEx(image, opening_image, alchemy::MORP_OPEN, {5, 5});
    alchemy::morphEx(image, closing_image, alchemy::MORP_CLOSE, {5, 5});
    alchemy::morphEx(image, th_image, alchemy::MORP_TOPHAT, {5, 5});
    alchemy::morphEx(image, b_image, alchemy::MORP_BLACKHAT, {5, 5});

    alchemy::imshow("原图", image);
    alchemy::imshow("腐蚀", erode_image);
    alchemy::imshow("膨胀", dilate_image);
    alchemy::imshow("开运算", opening_image);
    alchemy::imshow("闭运算", closing_image);
    alchemy::imshow("顶帽", th_image);
    alchemy::imshow("黑帽", b_image);
    alchemy::waitKey(0);

    return 0;
}