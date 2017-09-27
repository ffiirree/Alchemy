#include "zmatrix.h"
#include "zcore/zdef.h"

int main(int argc, char *argv[])
{
    // 载入原始图像
    auto image = z::imread("test.jpeg");
    z::Matrix gray;
    z::cvtColor(image, gray, z::BGR2GRAY);
    z::medianFilter(gray, gray, z::Size(3, 3));

    // 二值化
    auto bin_image = gray > 175;
    //auto bin_image_2 = gray > 175;


    // 寻找轮廓
    std::vector<std::vector<z::Point>> contours;
    auto res = z::Matrix::zeros(image.rows, image.cols, 3);
    z::findContours(bin_image, contours);

    // 显示结果
    uint8_t r = 50, g = 100, b = 150;
    for (const auto &c : contours) {
        for (const auto &j : c) {
            res.at<z::Vec3u8>(j.x, j.y) = { b, g, r };
        }
        r += 25, b += 50, b += 75;
    }

    //// 寻找最外轮廓
    //contours.clear();
    //z::findOutermostContours(bin_image_2, contours);

    //// 显示结果
    //res.zeros();
    //for (const auto &c : contours) {
    //    for (const auto &j : c) {
    //        *res.ptr<z::Vec3u8>(j.x, j.y) = { b, g, r };
    //    }
    //    r += 25, b += 50, b += 75;
    //}
    //cv::imshow("findOutermostContours", cv::Mat(res));

    //cv::waitKey(0);
    return 0;
}
