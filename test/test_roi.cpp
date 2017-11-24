#include "zmatrix.h"

int main()
{
    auto image = z::imread("test.jpeg");
    auto logo = z::imread("logo.jpeg");

    auto roi = image(z::Rect(50, 50, logo.cols, logo.rows));

    z::addWeighted(roi, 0.2, logo, 0.7, 0.0, roi);
    // 测试
    roi += z::Scalar(50, 50, 50);

    z::imshow("原图像", image);
    z::imshow("log", logo);
    z::imshow("roi:(50, 50, 200, 200) += (50, 50, 50)", roi);
    z::waitKey(0);
    return 0;
}