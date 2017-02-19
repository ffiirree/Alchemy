#include <iostream>
#include <opencv2\opencv.hpp>

#include "zcore\zcore.h"
#include "zimgproc\zimgproc.h"
#include "zimgproc\transform.h"
#include "zgui\zgui.h"
#include "zcore\debug.h"

int main(int argc, char *argv[])
{
    // 载入原始图像
    z::Matrix8u test = z::imread("test.jpeg");
    cv::imshow("origin image", cv::Mat(test));

    // 灰度图
    z::Matrix8u gray(test.size(), 1);
    z::cvtColor(test, gray, BGR2GRAY);
    cv::imshow("gray", cv::Mat(gray));

    // 中值滤波
    z::medianFilter(gray, gray, z::Size(3, 3));

    // 二值化
    auto bin_image = gray > 150;
    auto bin_image_2 = gray > 150;
    cv::imshow("binary image", cv::Mat(bin_image));


    // 寻找轮廓
    TimeStamp timer;
    std::vector<std::vector<z::Point>> contours;
    timer.start();
    z::Matrix8u res(test.rows, test.cols, 3);
    res.zeros();
    z::findContours(bin_image, contours);
    std::cout << timer.runtime() << std::endl;

    // 显示结果
    uchar r = 75, g = 150, b = 225;
    for (const auto &c : contours) {
        for (const auto &j : c) {
            *((z::Scalar *)res.ptr(j.x, j.y)) = z::Scalar(r, g, b);
        }
        r += 25, b += 50, b += 75;
    }
    cv::imshow("findContours", cv::Mat(res));

    // 寻找最外轮廓
    contours.clear();
    timer.start();
    z::findOutermostContours(bin_image_2, contours);
    std::cout << timer.runtime() << std::endl;

    // 显示结果
    res.zeros();
    for (const auto &c : contours) {
        for (const auto &j : c) {
            *((z::Scalar *)res.ptr(j.x, j.y)) = z::Scalar(r, g, b);
        }
        r += 25, b += 50, b += 75;
    }
    cv::imshow("findOutermostContours", cv::Mat(res));

    cv::waitKey(10);
    return 0;
}