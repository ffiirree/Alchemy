#include <alchemy.h>

int main()
{
    auto image = alchemy::imread("test.jpeg");
    alchemy::Matrix gray;
    alchemy::cvtColor(image, gray, alchemy::BGR2GRAY);
    alchemy::medianFilter(gray, gray, alchemy::Size(3, 3));

    // 二值化
    auto bin_image = gray > 175;

    // 寻找轮廓
    std::vector<std::vector<alchemy::Point>> contours;
    auto res = alchemy::Matrix::zeros(image.rows_, image.cols_, 3);
    alchemy::findContours(bin_image, contours);

    uint8_t r = 50, g = 100, b = 150;
    for (const auto &c : contours) {
        for (const auto &j : c) {
            res.at<alchemy::Vec3u8>(j.x, j.y) = { b, g, r };
        }
        r += 25, b += 50, b += 75;
    }
    alchemy::imshow("1. findContours()=>所有轮廓", res);

    // 寻找最外轮廓
    contours.clear();
    alchemy::findOutermostContours(bin_image, contours);

    res = alchemy::Matrix::zeros(image.rows_, image.cols_, 3);
    for (const auto &c : contours) {
        for (const auto &j : c) {
            res.at<alchemy::Vec3u8>(j.x, j.y) = { b, g, r };
        }
        r += 25, b += 50, b += 75;
    }

    alchemy::imshow("2. findOutermostContours()=>最外轮廓", res);

    alchemy::waitKey(0);
    return 0;
}
