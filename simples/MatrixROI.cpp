#include <alchemy.h>

int main()
{
    auto image = alchemy::imread("../resources/test.jpeg");
    auto logo = alchemy::imread("../resources/logo.jpeg");

    auto roi = image(alchemy::Rect(50, 50, logo.cols_, logo.rows_));
    roi.fill(alchemy::Scalar(-5, 6, 7));

//    alchemy::addWeighted(roi, 0.2, logo, 0.7, 0.0, roi);
//    // 测试
//    roi += alchemy::Scalar(50, 50, 50);
//
    alchemy::imshow("原图像", image);
//    alchemy::imshow("log", logo);
    alchemy::imshow("roi:(50, 50, 200, 200) += (50, 50, 50)", roi);
    alchemy::waitKey(0);
    return 0;
}