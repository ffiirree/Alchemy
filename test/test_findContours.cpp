#include <iostream>
#include <opencv2\opencv.hpp>

#include "zmatrix.h"

int main(int argc, char *argv[])
{
    // 载入原始图像
    z::Matrix test = z::imread("test.jpeg");
    cv::imshow("origin image", cv::Mat(test));

    // 灰度图
    z::Matrix gray(test.size(), 1);
    z::cvtColor(test, gray, BGR2GRAY);
    cv::imshow("gray", cv::Mat(gray));

    // 中值滤波
    z::medianFilter(gray, gray, z::Size(3, 3));

    // 二值化
    z::Matrix bin_image = gray > 175;
    z::Matrix bin_image_2 = gray > 175;
    cv::imshow("binary image", cv::Mat(bin_image));


    // 寻找轮廓
    std::vector<std::vector<z::Point>> contours;
    z::Matrix res(test.rows, test.cols, 3);
    res.zeros();
    z::findContours(bin_image, contours);

    // 显示结果
    uchar r = 50, g = 100, b = 150;
    for (const auto &c : contours) {
        for (const auto &j : c) {
            *res.ptr<z::Vec3u8>(j.x, j.y) = { b, g, r };
        }
        r += 25, b += 50, b += 75;
    }
    cv::imshow("findContours", cv::Mat(res));

    // 寻找最外轮廓
    contours.clear();
    z::findOutermostContours(bin_image_2, contours);

    // 显示结果
    res.zeros();
    for (const auto &c : contours) {
        for (const auto &j : c) {
            *res.ptr<z::Vec3u8>(j.x, j.y) = { b, g, r };
        }
        r += 25, b += 50, b += 75;
    }
    cv::imshow("findOutermostContours", cv::Mat(res));

    cv::waitKey(0);
    return 0;
}