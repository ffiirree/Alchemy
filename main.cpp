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
    cv::imshow("binary image", cv::Mat(bin_image));

    // 寻找轮廓
    TimeStamp timer;
    std::vector<std::vector<z::Point>> contours;
    timer.start();
    z::Matrix8u res(test.rows, test.cols, 3);
    z::findOutermostContours(bin_image, contours);
    std::cout << timer.runtime() << std::endl;

    for (const auto &c : contours) {
        for (const auto &j : c) {
            res.ptr(j.x, j.y)[0] = 255;
            res.ptr(j.x, j.y)[1] = 0;
            res.ptr(j.x, j.y)[2] = 0;
        }
    }
    
    cv::imshow("res", cv::Mat(res));
    cv::waitKey(0);
	return 0;
}