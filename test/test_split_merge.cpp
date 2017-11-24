#include "zmatrix.h"

int main()
{
    auto image = z::imread("test.jpeg");
    z::imshow("原图", image);

    // 通道分离
    std::vector<z::Matrix> mv;
    z::spilt(image, mv);

    z::imshow("B", mv[0]);
    z::imshow("G", mv[1]);
    z::imshow("R", mv[2]);

    // 合成
    z::Matrix merge_image;
    z::merge(mv, merge_image);
    z::imshow("再次合成", merge_image);

    z::waitKey(0);

    return 0;
}